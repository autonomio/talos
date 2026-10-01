"""Verified, bounded transports for explicitly selected remote entropy services."""

import json
import math
import os
import re
import ssl
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from talos.utils.exceptions import TalosDataError

__all__ = ['sample_indexes']

_ANU_URL = 'https://api.quantumnumbers.anu.edu.au'
_RANDOM_ORG_URL = 'https://api.random.org/json-rpc/4/invoke'
_BATCH_SIZE = 1024
_TIMEOUT_SECONDS = 15
_RESPONSE_LIMIT = 1_048_576


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int, msg: str,
                         headers: object, newurl: str) -> None:
        """Refuse redirects before credentials could reach another origin or protocol."""
        del req, fp, msg, headers, newurl
        raise TalosDataError(f'Entropy service redirect (HTTP {code}) is not supported.')


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TalosDataError('Entropy response must contain JSON objects.')
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TalosDataError('Entropy response object keys must be strings.')
    return {str(key): item for key, item in mapping.items()}


def _request_json(request: Request) -> dict[str, object]:
    if request.type != 'https':
        raise TalosDataError('Entropy requests require HTTPS.')
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    opener = build_opener(_NoRedirect(), HTTPSHandler(context=context))
    with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
        content = response.read(_RESPONSE_LIMIT + 1)
    if len(content) > _RESPONSE_LIMIT:
        raise TalosDataError('Entropy response exceeds the size limit.')
    try:
        value: object = json.loads(content)
    except (ValueError, UnicodeError):
        raise TalosDataError('Entropy response is not valid JSON.') from None
    return _mapping(value)


def _credential(variable: str, *, uuid_key: bool = False) -> str:
    path = os.environ.get(variable)
    if not path:
        raise TalosDataError(f'Set {variable} to a caller-owned API key file.')
    try:
        with Path(path).open(encoding='utf-8') as source:
            content = source.read(4098)
        key = content.strip()
    except (OSError, UnicodeError):
        raise TalosDataError(f'{variable} must identify a readable UTF-8 API key file.') from None
    if len(content) > 4097 or not key or len(key) > 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise TalosDataError(f'{variable} must contain one nonempty ASCII API key.')
    if uuid_key and re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', key) is None:
        raise TalosDataError(f'{variable} must contain a UUID API key.')
    return key


def _integers(value: object, length: int, maximum: int) -> list[int]:
    if not isinstance(value, list):
        raise TalosDataError('Entropy response must contain an integer list.')
    values = cast(list[object], value)
    if len(values) != length:
        raise TalosDataError('Entropy response has the wrong number of integers.')
    result: list[int] = []
    for item in values:
        if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= maximum:
            raise TalosDataError('Entropy response contains an invalid integer.')
        result.append(item)
    return result


def _quantum_words() -> Iterator[int]:
    while True:
        key = _credential('TALOS_ANU_KEY_FILE')
        request = Request(f'{_ANU_URL}?length={_BATCH_SIZE}&type=uint16', headers={'x-api-key': key})
        response = _request_json(request)
        if response.get('success') is not True or response.get('type', 'uint16') != 'uint16' or response.get('length', _BATCH_SIZE) != _BATCH_SIZE:
            raise TalosDataError('ANU returned an invalid uint16 response.')
        yield from _integers(response.get('data'), _BATCH_SIZE, 65535)


def _quantum_indexes(maximum: int, count: int) -> list[int]:
    """Retain the legacy uint16-to-index mapping while bounding rejected draws."""
    words = _quantum_words()
    width = math.ceil(math.ceil(math.log(maximum + 1, 2)) / 16)
    modulus = (2 ** (width * 16) - 1) / maximum
    ceiling = modulus * maximum
    result: list[int] = []
    for _ in range(count):
        for _attempt in range(32):
            number = 0
            for _word in range(width):
                number = (number << 16) + next(words)
            if number < ceiling:
                result.append(int(number / modulus))
                break
        else:
            raise TalosDataError('ANU did not supply admissible entropy after 32 draws.')
    return result


def _ambience_indexes(maximum: int, count: int) -> list[int]:
    if maximum > 1_000_000_000:
        raise TalosDataError('RANDOM.ORG supports maximum indexes up to 1000000000.')
    result: list[int] = []
    while len(result) < count:
        length = min(count - len(result), 10000)
        key = _credential('TALOS_RANDOM_ORG_KEY_FILE', uuid_key=True)
        body = {'jsonrpc': '2.0', 'method': 'generateIntegerSequences', 'id': 1418,
                'params': {'apiKey': key, 'n': 1, 'length': length, 'min': 0,
                           'max': maximum, 'replacement': True, 'base': 10}}
        response = _request_json(Request(_RANDOM_ORG_URL, data=json.dumps(body).encode(),
                                        headers={'Content-Type': 'application/json'}))
        if response.get('jsonrpc') != '2.0' or response.get('id') != 1418 or 'error' in response:
            raise TalosDataError('RANDOM.ORG rejected the entropy request.')
        values = _mapping(response.get('result'))
        data = _mapping(values.get('random')).get('data')
        if not isinstance(data, list):
            raise TalosDataError('RANDOM.ORG must return integer sequences.')
        sequences = cast(list[object], data)
        if len(sequences) != 1:
            raise TalosDataError('RANDOM.ORG must return one integer sequence.')
        result.extend(_integers(sequences[0], length, maximum))
        delay = values.get('advisoryDelay', 0)
        if not isinstance(delay, int) or isinstance(delay, bool) or not 0 <= delay <= 60000:
            raise TalosDataError('RANDOM.ORG advisory delay must be between 0 and 60000 milliseconds.')
        if delay:
            time.sleep(delay / 1000)
    return result


def sample_indexes(maximum: int, count: int, method: str) -> list[int]:
    """Fetch provider candidates; the caller retains unique legal index selection."""
    if type(maximum) is not int or type(count) is not int or not 1 <= count <= maximum:
        raise TalosDataError('Remote sampling requires positive integer count <= maximum.')
    if method == 'quantum':
        return _quantum_indexes(maximum, count)
    if method == 'ambience':
        return _ambience_indexes(maximum, count)
    raise ValueError(f'Unknown remote entropy method: {method!r}')
