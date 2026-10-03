import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const braces = require('braces');
const Policy = require('http-cache-semantics');

for (const method of ['compile', 'expand', 'stringify']) {
  for (const [name, pattern] of [
    ['braces', '{'.repeat(4000)+'a'+'}'.repeat(4000)],
    ['unclosed braces', '{'.repeat(4000)+'a'],
    ['parentheses', '('.repeat(4000)+'a'+')'.repeat(4000)],
    ['mixed nesting', '{('.repeat(2000)+'a'+')}'.repeat(2000)],
    ['dollar braces', '${'.repeat(2000)+'a'+'}'.repeat(2000)],
  ]) {
    test(`${method} rejects deep ${name} before stack exhaustion`, () => {
      assert.throws(() => braces[method](pattern), e => e instanceof SyntaxError && /nesting/.test(e.message));
    });
  }
  test(`${method} rejects deep caller-created ASTs`, () => {
    let ast = {type:'text',value:'a'};
    for (let i=0;i<4000;i++) ast = {type:'root', nodes:[ast]};
    assert.throws(() => braces[method](ast), e => e instanceof SyntaxError && /nesting/.test(e.message));
  });
  test(`${method} rejects a cyclic AST`, () => {
    const ast = {type:'root',nodes:[]};
    ast.nodes.push(ast);
    assert.throws(() => braces[method](ast), e => e instanceof SyntaxError && /nesting/.test(e.message));
  });
}

test('ordinary nested expansion, ranges and escaping retain semantics', () => {
  assert.deepEqual(braces.expand('foo/{a,{b,c}}/{1..2}'), ['foo/a/1','foo/a/2','foo/b/1','foo/b/2','foo/c/1','foo/c/2']);
  assert.equal(braces.compile('foo/{a,b}'), 'foo/(a|b)');
  assert.equal(braces.stringify('foo/{a,b}'), 'foo/{a,b}');
  assert.equal(braces.stringify('\\{literal\\}'), '{literal}');
  assert.deepEqual(braces.expand('[{a,b}]'), ['[{a,b}]']);
  assert.deepEqual(braces.expand('"{a,b}"'), ['{a,b}']);
  assert.equal(braces.compile('{'.repeat(100)+'a'+'}'.repeat(100)).length,201);
});

const request = (headers={}) => ({url:'https://example.invalid/account',method:'GET',headers:{host:'example.invalid',...headers}});
for (const directive of ['max-stale','max-stale=1000000']) {
  for (const [name, originalHeaders, responseHeaders, options] of [
    ['session cookie', {}, {'set-cookie':'session=USER_A; Secure; HttpOnly'}, {}],
    ['private response', {}, {'cache-control':'private, max-age=0'}, {}],
    ['no-store response', {}, {'cache-control':'no-store'}, {}],
    ['authenticated response', {authorization:'Bearer fake-user-a'}, {'cache-control':'max-age=1000'}, {}],
    ['no-cache response', {}, {'cache-control':'no-cache, max-age=0'}, {}],
    ['request no-store', {'cache-control':'no-store'}, {'cache-control':'max-age=10'}, {}],
    ['proxy revalidation', {}, {'cache-control':'public, proxy-revalidate, max-age=0'}, {}],
    ['vary-star response', {}, {'vary':'*','cache-control':'public, max-age=0'}, {}],
  ]) {
    test(`${directive} cannot revive ${name} for another user`, () => {
      const policy = new Policy(request(originalHeaders), {status:200,headers:responseHeaders}, options);
      policy.now = () => policy._responseTime + 10000;
      const outcome = policy.evaluateRequest(request({'cache-control':directive}));
      assert.equal(outcome.response, undefined);
      assert.equal(outcome.revalidation.synchronous, true);
      assert.equal(policy.satisfiesWithoutRevalidation(request({'cache-control':directive})),false);
      // Disk-cache serialization must retain the same guard.
      const restored = Policy.fromObject(policy.toObject());
      restored.now = policy.now;
      assert.equal(restored.evaluateRequest(request({'cache-control':directive})).response,undefined);
    });
  }
}

test('public stale content still obeys max-stale and fresh cache hits work', () => {
  const policy = new Policy(request(), {status:200,headers:{'cache-control':'public, max-age=5'}});
  policy.now = () => policy._responseTime + 10000;
  assert.ok(policy.evaluateRequest(request({'cache-control':'max-stale=6'})).response);
  assert.equal(policy.evaluateRequest(request({'cache-control':'max-stale=4'})).response,undefined);
  policy.now = () => policy._responseTime + 1000;
  assert.ok(policy.evaluateRequest(request()).response);
});

test('non-shared private cache retains explicit session-cookie semantics', () => {
  const policy = new Policy(request(), {status:200,headers:{'cache-control':'private, max-age=5','set-cookie':'session=USER_A'}}, {shared:false});
  policy.now = () => policy._responseTime + 10000;
  assert.ok(policy.evaluateRequest(request({'cache-control':'max-stale'})).response);
});
