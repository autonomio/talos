"""Loopback HTTPS contract tests, without accounts or physical entropy claims."""

import json
import ssl
import subprocess
import threading
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from talos.reducers import remote_entropy
from talos.reducers.sample_reducer import sample_reducer
from talos.utils.exceptions import TalosDataError

_ANU_KEY = 'loopback-only-control-key'
_RANDOM_KEY = '00000000-0000-0000-0000-000000000001'


@pytest.fixture(scope='module')
def certificate(tmp_path_factory):
    root = tmp_path_factory.mktemp('entropy-tls')
    cert, key = root / 'cert.pem', root / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'], check=True)
    return cert, key


class _ProviderHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._respond()

    def do_POST(self):
        self._respond()

    def _respond(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.server.observed.append({'method': self.command, 'path': self.path,
            'headers': dict(self.headers), 'body': body,
            'tls': self.connection.version() if isinstance(self.connection, ssl.SSLSocket) else None})
        status, headers, data = self.server.response_for(self.server.observed)
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def provider(response_for, certificate=None, tls_version=ssl.TLSVersion.TLSv1_2):
    server = ThreadingHTTPServer(('127.0.0.1', 0), _ProviderHandler)
    server.observed = []
    server.response_for = response_for
    if certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = tls_version
        context.maximum_version = tls_version
        if tls_version < ssl.TLSVersion.TLSv1_2:
            context.set_ciphers('ALL:@SECLEVEL=0')
        context.load_cert_chain(*(str(path) for path in certificate))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02})
    thread.start()
    scheme = 'https' if certificate is not None else 'http'
    with ExitStack() as cleanup:
        cleanup.callback(_join_provider, thread)
        cleanup.callback(server.server_close)
        cleanup.callback(server.shutdown)
        yield server, f'{scheme}://localhost:{server.server_port}/'


def _join_provider(thread):
    thread.join(timeout=3)
    assert not thread.is_alive()


def json_response(payload):
    return 200, {'Content-Type': 'application/json'}, json.dumps(payload).encode()


def credentials(tmp_path, monkeypatch):
    for variable, key in (('TALOS_ANU_KEY_FILE', _ANU_KEY), ('TALOS_RANDOM_ORG_KEY_FILE', _RANDOM_KEY)):
        path = tmp_path / variable
        path.write_text(key + '\n')
        monkeypatch.setenv(variable, str(path))


def endpoint(monkeypatch, method, url):
    monkeypatch.setattr(remote_entropy, '_ANU_URL' if method == 'quantum' else '_RANDOM_ORG_URL', url)


def valid_response(method, observations):
    if method == 'quantum':
        prefix = [0, 0, 65534] if len(observations) == 1 else [21845]
        return json_response({'type': 'uint16', 'length': 1024, 'success': True,
                              'data': prefix + [0] * (1024 - len(prefix))})
    return json_response({'jsonrpc': '2.0', 'id': 1418,
        'result': {'random': {'data': [[0, 0, 3] if len(observations) == 1 else [1, 2]]}, 'advisoryDelay': 0}})


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
@pytest.mark.parametrize('version', [ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3])
def test_verified_transport_preserves_unique_legal_candidates(method, version, certificate, tmp_path, monkeypatch, capsys):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: valid_response(method, observations), certificate, version) as (server, url):
        endpoint(monkeypatch, method, url)
        assert sample_reducer(3, 3, method) == ([0, 2, 1] if method == 'quantum' else [0, 1, 2])
        assert len(server.observed) == 2
        assert {item['tls'] for item in server.observed} == {version.name.replace('_', '.')}
        for index, item in enumerate(server.observed):
            assert _ANU_KEY not in item['path'] and _RANDOM_KEY not in item['path']
            if method == 'quantum':
                assert item['method'] == 'GET' and item['body'] == b''
                assert item['headers']['X-Api-Key'] == _ANU_KEY
                assert parse_qs(urlsplit(item['path']).query) == {'length': ['1024'], 'type': ['uint16']}
            else:
                body = json.loads(item['body'])
                assert item['method'] == 'POST' and item['headers']['Content-Type'] == 'application/json'
                assert body['method'] == 'generateIntegerSequences' and body['id'] == 1418
                assert body['params'] == {'apiKey': _RANDOM_KEY, 'n': 1,
                    'length': 3 if index == 0 else 2,
                    'min': 0, 'max': 3, 'replacement': True, 'base': 10}
    captured = capsys.readouterr()
    assert _ANU_KEY not in captured.out + captured.err and _RANDOM_KEY not in captured.out + captured.err


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
@pytest.mark.parametrize('failure', ['untrusted', 'hostname'])
def test_invalid_certificate_is_rejected_before_private_headers_or_body(method, failure, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    if failure == 'hostname':
        monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    else:
        monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    with provider(lambda observations: valid_response(method, observations), certificate) as (server, url):
        endpoint(monkeypatch, method, url.replace('localhost', '127.0.0.1') if failure == 'hostname' else url)
        with pytest.raises(URLError) as error:
            sample_reducer(1, 3, method)
        assert isinstance(error.value.reason, ssl.SSLCertVerificationError)
        assert server.observed == []
        assert _ANU_KEY not in str(error.value) and _RANDOM_KEY not in str(error.value)


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
@pytest.mark.parametrize('status', [302, 307])
def test_redirect_cannot_send_credentials_to_plain_http(method, status, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response({})) as (sink, insecure_url):
        with provider(lambda observations: (status, {'Location': insecure_url}, b''), certificate) as (server, url):
            endpoint(monkeypatch, method, url)
            with pytest.raises((TalosDataError, HTTPError)) as error:
                sample_reducer(1, 3, method)
            assert len(server.observed) == 1 and sink.observed == []
            assert _ANU_KEY not in str(error.value) and _RANDOM_KEY not in str(error.value)


@pytest.mark.parametrize('method,variable', [('quantum', 'TALOS_ANU_KEY_FILE'), ('ambience', 'TALOS_RANDOM_ORG_KEY_FILE')])
def test_credentials_rotate_without_code_change(method, variable, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))

    def response(observations):
        if method == 'quantum':
            return json_response({'type': 'uint16', 'length': 1024, 'success': True, 'data': [0] * 1024})
        return json_response({'jsonrpc': '2.0', 'id': 1418, 'result': {'random': {'data': [[0]]}, 'advisoryDelay': 0}})
    with provider(response, certificate) as (server, url):
        endpoint(monkeypatch, method, url)
        assert sample_reducer(1, 3, method) == [0]
        path = tmp_path / variable
        replacement = 'replacement-loopback-control-key' if method == 'quantum' else '00000000-0000-0000-0000-000000000002'
        path.write_text(replacement + '\n')
        assert sample_reducer(1, 3, method) == [0]
        sent = [item['headers']['X-Api-Key'] if method == 'quantum' else json.loads(item['body'])['params']['apiKey'] for item in server.observed]
        assert sent == ([_ANU_KEY, replacement] if method == 'quantum' else [_RANDOM_KEY, replacement])


@pytest.mark.parametrize('method,variable', [('quantum', 'TALOS_ANU_KEY_FILE'), ('ambience', 'TALOS_RANDOM_ORG_KEY_FILE')])
@pytest.mark.parametrize('invalid', [None, '', 'line\nbreak', 'é', 'x' * 5000])
def test_invalid_credentials_fail_before_transport(method, variable, invalid, certificate, tmp_path, monkeypatch, capsys):
    credentials(tmp_path, monkeypatch)
    if invalid is None:
        monkeypatch.delenv(variable)
    else:
        (tmp_path / variable).write_text(invalid)
    with provider(lambda observations: valid_response(method, observations), certificate) as (server, url):
        endpoint(monkeypatch, method, url)
        with pytest.raises(TalosDataError, match=variable):
            sample_reducer(1, 3, method)
        assert server.observed == []
    captured = capsys.readouterr()
    assert _ANU_KEY not in captured.out + captured.err and _RANDOM_KEY not in captured.out + captured.err


@pytest.mark.parametrize('invalid', [True, -1, 65536, '3', 1.5])
def test_quantum_rejects_malformed_provider_integers(invalid, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response({'type': 'uint16', 'length': 1024,
            'success': True, 'data': [invalid] + [0] * 1023}), certificate) as (server, url):
        endpoint(monkeypatch, 'quantum', url)
        with pytest.raises(TalosDataError, match='invalid integer'):
            sample_reducer(1, 3, 'quantum')
        assert len(server.observed) == 1


@pytest.mark.parametrize('invalid', [True, -1, 4, '3', 1.5])
def test_ambience_rejects_malformed_provider_integers(invalid, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response({'jsonrpc': '2.0', 'id': 1418,
            'result': {'random': {'data': [[invalid]]}, 'advisoryDelay': 0}}), certificate) as (server, url):
        endpoint(monkeypatch, 'ambience', url)
        with pytest.raises(TalosDataError, match='invalid integer'):
            sample_reducer(1, 3, 'ambience')
        assert len(server.observed) == 1


def test_quantum_bounds_unusable_raw_entropy(certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response({'type': 'uint16', 'length': 1024,
            'success': True, 'data': [65535] * 1024}), certificate) as (server, url):
        endpoint(monkeypatch, 'quantum', url)
        with pytest.raises(TalosDataError, match='32 draws'):
            sample_reducer(1, 3, 'quantum')
        assert len(server.observed) == 1


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
def test_tls_before_12_is_rejected(method, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with pytest.warns(DeprecationWarning):
        with provider(lambda observations: valid_response(method, observations), certificate, ssl.TLSVersion.TLSv1_1) as (server, url):
            endpoint(monkeypatch, method, url)
            with pytest.raises(URLError) as error:
                sample_reducer(1, 3, method)
            assert isinstance(error.value.reason, ssl.SSLError)
            assert server.observed == []


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
def test_https_redirect_is_also_refused(method, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response({}), certificate) as (sink, next_url):
        with provider(lambda observations: (302, {'Location': next_url}, b''), certificate) as (server, url):
            endpoint(monkeypatch, method, url)
            with pytest.raises(TalosDataError, match='redirect'):
                sample_reducer(1, 3, method)
            assert len(server.observed) == 1 and sink.observed == []


@pytest.mark.parametrize('body', [b'{', b'\xff', b'[]', b'0' * 1_048_578])
def test_invalid_and_oversized_json_responses_are_rejected(body, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: (200, {}, body), certificate) as (server, url):
        endpoint(monkeypatch, 'quantum', url)
        with pytest.raises(TalosDataError):
            sample_reducer(1, 3, 'quantum')
        assert len(server.observed) == 1


@pytest.mark.parametrize('changes', [{'success': False}, {'type': 'uint8'}, {'length': 1}, {'data': [0]}])
def test_quantum_checks_entire_response_contract(changes, certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    payload = {'type': 'uint16', 'length': 1024, 'success': True, 'data': [0] * 1024, **changes}
    with provider(lambda observations: json_response(payload), certificate) as (server, url):
        endpoint(monkeypatch, 'quantum', url)
        with pytest.raises(TalosDataError):
            sample_reducer(1, 3, 'quantum')
        assert len(server.observed) == 1


def test_quantum_accepts_documented_success_data_response(certificate, tmp_path, monkeypatch):
    # The current official linked example reads success/data; metadata remains optional.
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response({'success': True, 'data': [0] * 1024}), certificate) as (server, url):
        endpoint(monkeypatch, 'quantum', url)
        assert sample_reducer(1, 3, 'quantum') == [0]
        assert len(server.observed) == 1


@pytest.mark.parametrize('payload', [
    {'jsonrpc': '2.0', 'id': 1418, 'error': {'message': _RANDOM_KEY, 'data': _RANDOM_KEY}},
    {'jsonrpc': '2.0', 'id': 1419},
    {'jsonrpc': '1.0', 'id': 1418},
    {'jsonrpc': '2.0', 'id': 1418, 'result': None},
    {'jsonrpc': '2.0', 'id': 1418, 'result': {'random': {'data': []}}},
    {'jsonrpc': '2.0', 'id': 1418, 'result': {'random': {'data': [[0]]}, 'advisoryDelay': True}},
    {'jsonrpc': '2.0', 'id': 1418, 'result': {'random': {'data': [[0]]}, 'advisoryDelay': -1}},
    {'jsonrpc': '2.0', 'id': 1418, 'result': {'random': {'data': [[0]]}, 'advisoryDelay': 60001}},
])
def test_ambience_checks_response_identity_shape_and_delay_without_echoing_key(payload, certificate, tmp_path, monkeypatch, capsys):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    with provider(lambda observations: json_response(payload), certificate) as (server, url):
        endpoint(monkeypatch, 'ambience', url)
        with pytest.raises(TalosDataError) as error:
            sample_reducer(1, 3, 'ambience')
        assert len(server.observed) == 1 and _RANDOM_KEY not in str(error.value)
    captured = capsys.readouterr()
    assert _RANDOM_KEY not in captured.out + captured.err


def test_ambience_batches_with_provider_limits_and_reloads_credential(certificate, tmp_path, monkeypatch):
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    replacement = '00000000-0000-0000-0000-000000000002'

    def response(observations):
        if len(observations) == 1:
            values = list(range(10000))
            (tmp_path / 'TALOS_RANDOM_ORG_KEY_FILE').write_text(replacement)
        else:
            values = [10000]
        return json_response({'jsonrpc': '2.0', 'id': 1418,
            'result': {'random': {'data': [values]}, 'advisoryDelay': 0}})
    with provider(response, certificate) as (server, url):
        endpoint(monkeypatch, 'ambience', url)
        assert sample_reducer(10001, 10001, 'ambience') == list(range(10001))
        sent = [json.loads(item['body'])['params'] for item in server.observed]
        assert [(item['length'], item['apiKey']) for item in sent] == [(10000, _RANDOM_KEY), (1, replacement)]


def test_ambience_provider_range_limit_fails_before_transport(monkeypatch):
    monkeypatch.delenv('TALOS_RANDOM_ORG_KEY_FILE', raising=False)
    with pytest.raises(TalosDataError, match='1000000000'):
        sample_reducer(1, 1_000_000_001, 'ambience')


@pytest.mark.parametrize('maximum,expected', [
    (31, [0, 0, 7, 15, 23, 30]),
    (65536, [0, 16384, 49152, 65535, 0, 16384]),
    (4294967296, [1, 2147532800, 4294901762, 81920, 3221291006, 131072]),
])
def test_quantum_keeps_recorded_legacy_uint16_mapping(maximum, expected, certificate, tmp_path, monkeypatch):
    # Recorded from Chances0.1.9 using this explicit generator; no remote randomness claim.
    credentials(tmp_path, monkeypatch)
    monkeypatch.setenv('SSL_CERT_FILE', str(certificate[0]))
    words = [0, 1, 16384, 32768, 49152, 65534, 65535, 2] * 128
    with provider(lambda observations: json_response({'type': 'uint16', 'length': 1024,
            'success': True, 'data': words}), certificate) as (server, url):
        endpoint(monkeypatch, 'quantum', url)
        assert remote_entropy.sample_indexes(maximum, 6, 'quantum') == expected
        assert len(server.observed) == 1


@pytest.mark.parametrize('method,variable', [('quantum', 'TALOS_ANU_KEY_FILE'), ('ambience', 'TALOS_RANDOM_ORG_KEY_FILE')])
@pytest.mark.parametrize('failure', ['missing_file', 'invalid_utf8'])
def test_credential_read_errors_do_not_expose_filename_or_contents(method, variable, failure, tmp_path, monkeypatch):
    path = tmp_path / _ANU_KEY
    if failure == 'invalid_utf8':
        path.write_bytes(_ANU_KEY.encode() + b'\xff')
    monkeypatch.setenv(variable, str(path))
    with pytest.raises(TalosDataError) as error:
        sample_reducer(1, 3, method)
    assert variable in str(error.value) and _ANU_KEY not in str(error.value)


def test_ambience_rejects_non_uuid_key_without_echoing_it(tmp_path, monkeypatch):
    path = tmp_path / 'key'
    path.write_text(_ANU_KEY)
    monkeypatch.setenv('TALOS_RANDOM_ORG_KEY_FILE', str(path))
    with pytest.raises(TalosDataError, match='UUID') as error:
        sample_reducer(1, 3, 'ambience')
    assert _ANU_KEY not in str(error.value)


@pytest.mark.parametrize('maximum,count', [(0, 1), (3, 0), (3, 4), (True, 1), (3, True)])
def test_remote_domain_is_validated_before_service_configuration(maximum, count):
    with pytest.raises(TalosDataError, match='positive integer'):
        remote_entropy.sample_indexes(maximum, count, 'quantum')
