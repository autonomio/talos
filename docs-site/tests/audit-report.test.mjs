import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';

import {
  DEFAULT_FLOOR,
  RELAXED_FLOOR,
  RELAXED_ROOTS,
  auditFailure,
} from '../scripts/audit-report.mjs';
import {productionRoots, rootsFrom} from '../scripts/audit-scope.mjs';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const NO_ROOTS = new Map();
const reached = (pkg, ...roots) => new Map([[pkg, new Set(roots)]]);
const report = (severity) => ({vulnerabilities: {victim: {severity}}});

test('fails closed on npm audit errors and malformed reports', () => {
  assert.match(
    auditFailure({error: {code: 'ENOAUDIT'}}, NO_ROOTS),
    /npm audit failed.*ENOAUDIT/
  );
  assert.equal(
    auditFailure({metadata: {}}, NO_ROOTS),
    'npm audit report has no vulnerabilities object'
  );
  assert.equal(auditFailure({vulnerabilities: {}}, NO_ROOTS), null);
});

test('a severity npm does not define fails loud rather than passing', () => {
  assert.match(
    auditFailure(report('catastrophic'), NO_ROOTS),
    /unknown severity "catastrophic" for victim/
  );
});

test('high outside the relaxed scope blocks', () => {
  assert.match(
    auditFailure(report('high'), reached('victim', 'react')),
    /victim \(high, floor info, via react\)/
  );
});

test('moderate advisories also block', () => {
  assert.match(auditFailure(report('moderate'), reached('victim', 'react')), /victim \(moderate, floor info/);
});

test('Docusaurus high advisories block without a blanket exemption', () => {
  assert.match(
    auditFailure(report('high'), reached('victim', '@docusaurus/core')),
    /victim \(high, floor info/
  );
});

test('critical in the relaxed scope still blocks', () => {
  assert.match(
    auditFailure(report('critical'), reached('victim', '@docusaurus/core')),
    /victim \(critical, floor info, via @docusaurus\/core\)/
  );
});

test('a package reached by a strict root stays strict', () => {
  // Shared between docusaurus and react: one relaxed parent must not lower the
  // floor for the path that arrives through react.
  assert.match(
    auditFailure(report('high'), reached('victim', '@docusaurus/core', 'react')),
    /victim \(high, floor info, via @docusaurus\/core, react\)/
  );
});

test('a package the lockfile walk cannot place keeps the default floor', () => {
  assert.match(
    auditFailure(report('high'), NO_ROOTS),
    /victim \(high, floor info, via unresolved\)/
  );
});

test('all production roots use the same zero-advisory policy', () => {
  assert.equal(DEFAULT_FLOOR, 'info');
  assert.equal(RELAXED_FLOOR, 'info');
  assert.deepEqual(RELAXED_ROOTS, []);
});

test('rootsFrom follows npm nesting rather than assuming a flat tree', () => {
  const packages = {
    'node_modules/root-a': {dependencies: {shared: '1'}},
    'node_modules/root-b': {dependencies: {shared: '1'}},
    // root-b resolves its own nested copy before the hoisted one.
    'node_modules/root-b/node_modules/shared': {dependencies: {deep: '1'}},
    'node_modules/shared': {},
    'node_modules/deep': {},
  };
  const roots = rootsFrom(packages, ['root-a', 'root-b']);
  assert.deepEqual([...roots.get('shared')].sort(), ['root-a', 'root-b']);
  // `deep` hangs off root-b's nested copy only.
  assert.deepEqual([...roots.get('deep')], ['root-b']);
});

test('a severity colliding with Object.prototype fails loud', () => {
  // `constructor` and `__proto__` resolve to inherited members on an object
  // literal, which would slip past the unknown-severity guard and then rank as
  // harmless against the floor.
  for (const severity of ['constructor', 'toString', '__proto__', 'valueOf']) {
    assert.match(
      auditFailure(report(severity), reached('victim', 'react')),
      /unknown severity/,
      `severity ${severity} did not fail loud`
    );
  }
});

test('rootsFrom refuses a production root the lockfile cannot place', () => {
  // Silently dropping it would relax: packages shared with the docusaurus stack
  // would keep docusaurus-only attribution and fall to the critical floor.
  assert.throws(
    () => rootsFrom({'node_modules/present': {}}, ['present', 'absent']),
    /production dependency 'absent'.*no entry in package-lock\.json/s
  );
});

test('rootsFrom walks optional and peer edges, not dependencies alone', () => {
  const packages = {
    'node_modules/strict': {optionalDependencies: {shared: '1'}},
    'node_modules/relaxed': {dependencies: {shared: '1'}},
    'node_modules/peered': {peerDependencies: {shared: '1'}},
    'node_modules/shared': {},
  };
  const roots = rootsFrom(packages, ['strict', 'relaxed', 'peered']);
  // Walking `dependencies` alone would attribute `shared` to `relaxed` only.
  assert.deepEqual(
    [...roots.get('shared')].sort(),
    ['peered', 'relaxed', 'strict']
  );
});

test('productionRoots places transitive packages, not just the roots themselves', () => {
  const roots = productionRoots(siteRoot);
  const manifest = JSON.parse(readFileSync(path.join(siteRoot, 'package.json'), 'utf8'));
  const direct = Object.keys(manifest.dependencies);

  for (const root of direct) {
    assert.ok(roots.has(root), `${root} is not placed`);
  }
  // The map must be dominated by packages that are not direct dependencies,
  // otherwise the walk stopped at the roots and every transitive advisory would
  // be attributed to nothing.
  const transitive = [...roots.keys()].filter((name) => !direct.includes(name));
  assert.ok(transitive.length > 100, `only ${transitive.length} transitive packages placed`);

  // At least one package must be owned by a root outside the relaxed set. If no
  // strict root ever resolves, every shared package silently relaxes.
  const strictlyOwned = [...roots.values()]
    .filter((owners) => [...owners].some((owner) => !RELAXED_ROOTS.includes(owner)));
  assert.ok(strictlyOwned.length > 0, 'no package is attributed to a strict root');

  // Every recorded owner must be a declared production dependency.
  for (const [name, owners] of roots) {
    assert.ok(owners.size > 0, `${name} has an empty owner set`);
    for (const owner of owners) {
      assert.ok(direct.includes(owner), `${name} names a non-root owner ${owner}`);
    }
  }
});
