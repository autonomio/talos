import assert from 'node:assert/strict';
import test from 'node:test';

import {
  checkLink,
  attemptLink,
  MAX_ATTEMPTS,
  RETRYABLE_STATUS,
} from '../scripts/check-external-links.mjs';

// The link check runs inside the required `pr_checks_lint` gate and makes live
// requests to third-party hosts. Without a retry, one rate-limit response
// reddens a required check for a reason the author cannot influence -- and a
// gate that fails for reasons unrelated to the PR is one people learn to
// re-run rather than read.
//
// The attempt function is injected so these cases never touch the network:
// a test that depended on a real host would have the flakiness it is meant to
// fix.

function responder(statuses) {
  const remaining = [...statuses];
  const calls = {count: 0};
  const attempt = async () => {
    calls.count += 1;
    const next = remaining.shift();
    if (next instanceof Error) {
      throw next;
    }
    return next;
  };
  return {attempt, calls};
}

const noSleep = async () => {};

test('retries a rate-limited response and then succeeds', async () => {
  const {attempt, calls} = responder([429, 200]);
  await checkLink('https://example.test/a', attempt, noSleep);
  assert.equal(calls.count, 2, 'a 429 must be retried, not reported');
});

test('retries a server error and then succeeds', async () => {
  const {attempt, calls} = responder([503, 200]);
  await checkLink('https://example.test/b', attempt, noSleep);
  assert.equal(calls.count, 2);
});

test('retries a network-level failure', async () => {
  const {attempt, calls} = responder([new Error('ECONNRESET'), 200]);
  await checkLink('https://example.test/c', attempt, noSleep);
  assert.equal(calls.count, 2, 'a reset is the host, not the link');
});

test('does not retry a genuine broken link', async () => {
  const {attempt, calls} = responder([404, 200]);
  await assert.rejects(
    () => checkLink('https://example.test/gone', attempt, noSleep),
    /returned 404/,
  );
  assert.equal(
    calls.count,
    1,
    'retrying a 404 would turn a real broken link into a slow real broken link',
  );
});

test('gives up after the attempt ceiling', async () => {
  const {attempt, calls} = responder([429, 429, 429, 200]);
  await assert.rejects(
    () => checkLink('https://example.test/busy', attempt, noSleep),
    /returned 429/,
  );
  assert.equal(
    calls.count,
    MAX_ATTEMPTS,
    'an unbounded retry turns a dead host into a hung required check',
  );
});

test('a first-attempt success makes no further request', async () => {
  const {attempt, calls} = responder([200, 500]);
  await checkLink('https://example.test/ok', attempt, noSleep);
  assert.equal(calls.count, 1);
});

test('the retryable set covers only transient statuses', () => {
  for (const status of [408, 425, 429, 500, 502, 503, 504]) {
    assert.ok(RETRYABLE_STATUS.has(status), `${status} should be retryable`);
  }
  for (const status of [400, 401, 403, 404, 410, 451]) {
    assert.ok(
      !RETRYABLE_STATUS.has(status),
      `${status} names a wrong link, so retrying it hides the finding`,
    );
  }
});


test('HEAD rejection passes only after a real successful GET', async () => {
  const methods = [];
  const status = await attemptLink('https://example.com/page', async (_url, method) => {
    methods.push(method);
    return {statusCode: method === 'HEAD' ? 403 : 200, resume() {}};
  });
  assert.equal(status, 200);
  assert.deepEqual(methods, ['HEAD', 'GET']);
});

test('a missing page stays broken when GET confirms its missing status', async () => {
  const status = await attemptLink('https://example.com/missing', async () => ({statusCode: 404, resume() {}}));
  assert.equal(status, 404);
});
