import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {acceptedPackages, auditExecutionFailure, reviewedExceptions, verifyInstalledExceptions} from '../scripts/audit-exceptions.mjs';
import {auditFailure} from '../scripts/audit-report.mjs';

const entries = JSON.parse(readFileSync(new URL('../security-exceptions.json', import.meta.url), 'utf8'));
const NOW = new Date('2026-10-03T12:00:00Z');
const review = () => reviewedExceptions(structuredClone(entries), NOW);
const advisory = (entry) => ({
  name: entry.package, dependency: entry.package, severity: entry.severity,
  url: `https://github.com/advisories/${entry.id}`,
});
const finding = (name, via, severity = 'high') => ({name, severity, via, nodes: [`node_modules/${name}`]});
function fixture() {
  const vulnerabilities = Object.fromEntries(entries.map((entry) => [entry.package, finding(entry.package, [advisory(entry)])]));
  vulnerabilities.tool = finding('tool', entries.map((entry) => entry.package));
  vulnerabilities.root = finding('root', ['tool']);
  const packages = Object.fromEntries(entries.map((entry) => [`node_modules/${entry.package}`, {version: entry.version}]));
  packages['node_modules/tool'] = {version: '1.0.0'};
  packages['node_modules/root'] = {version: '1.0.0'};
  const roots = new Map(Object.keys(vulnerabilities).map((name) => [name, new Set(['root'])]));
  return {report: {vulnerabilities}, packages, roots};
}
function decision(candidate, reviewed = review()) {
  const accepted = acceptedPackages(candidate.report, candidate.packages, reviewed);
  return {accepted, failure: auditFailure(candidate.report, candidate.roots, accepted)};
}

test('checked-in approvals are exactly the two agreed IDs, versions and expiry', () => {
  assert.deepEqual(entries.map(({id, package: name, version, severity, approved_by, approved_on, expires}) =>
    [id, name, version, severity, approved_by, approved_on, expires]), [
    ['GHSA-vfj7-8cjw-p6xm', 'braces', '3.0.3', 'high', 'mikkokotila', '2026-10-03', '2026-11-02'],
    ['GHSA-ch52-4w7c-c8xp', 'http-cache-semantics', '4.2.0', 'high', 'mikkokotila', '2026-10-03', '2026-11-02'],
  ]);
  assert.equal(reviewedExceptions(entries, new Date('2026-11-01T23:59:59.999Z')).size, 2);
  assert.throws(() => reviewedExceptions(entries, new Date('2026-11-02T00:00:00Z')), /expired/);
});

test('malformed, duplicate, unreviewed, future and overlong approvals fail closed', () => {
  for (const mutation of [
    (entry) => { entry.id = 'other'; },
    (entry) => { entry.version = '^3.0.3'; },
    (entry) => { entry.reason = ' '; },
    (entry) => { entry.approved_by = ''; },
    (entry) => { entry.severity = 'unknown'; },
    (entry) => { entry.approved_on = '2026-10-04'; },
    (entry) => { entry.expires = '2026-11-03'; },
    (entry) => { entry.expires = '2026-10-03'; },
    (entry) => { entry.expires = '2026-02-30'; },
    (entry) => { delete entry.approved_on; },
  ]) {
    const changed = structuredClone(entries);
    mutation(changed[0]);
    assert.throws(() => reviewedExceptions(changed, NOW));
  }
  assert.throws(() => reviewedExceptions([entries[0], entries[0]], NOW), /Duplicate/);
  for (const invalid of [null, {}, 'invalid']) assert.throws(() => reviewedExceptions(invalid, NOW));
  assert.throws(() => reviewedExceptions(entries, new Date('invalid')), /audit time/);
});

test('complete transitive cause chains pass only with their exact reviewed leaves', () => {
  const candidate = fixture();
  const result = decision(candidate);
  assert.deepEqual([...result.accepted].sort(), ['braces', 'http-cache-semantics', 'root', 'tool']);
  assert.equal(result.failure, null);
  assert.match(decision(candidate, new Map()).failure, /braces/);
  assert.match(auditFailure(candidate.report, new Map(), result.accepted), /via unresolved/);
});

test('a new advisory on the accepted package blocks it and every affected parent', () => {
  const candidate = fixture();
  candidate.report.vulnerabilities.braces.via.push({...advisory(entries[0]), url: 'https://github.com/advisories/GHSA-abcd-abcd-abcd'});
  const result = decision(candidate);
  assert.deepEqual([...result.accepted], ['http-cache-semantics']);
  assert.match(result.failure, /braces/);
  assert.match(result.failure, /root/);
  assert.match(result.failure, /tool/);
});

test('advisory identity, severity and every affected lock node remain bound', () => {
  for (const mutation of [
    (value) => { value.via[0].name = 'different'; },
    (value) => { value.via[0].dependency = 'different'; },
    (value) => { value.via[0].url += '?accepted=true'; },
    (value) => { value.via[0].severity = 'critical'; },
    (value) => { value.severity = 'critical'; },
    (value) => { value.nodes = []; },
    (value) => { value.nodes.push('node_modules/unlocked'); },
    (value) => { value.nodes = ['node_modules/tool']; },
    (value) => { value.via = []; },
    (value) => { value.via = [null]; },
  ]) {
    const candidate = fixture();
    mutation(candidate.report.vulnerabilities.braces);
    assert.match(decision(candidate).failure, /braces/);
  }
  const changed = fixture();
  changed.packages['node_modules/braces'].version = '3.0.2';
  assert.match(decision(changed).failure, /braces/);
  const nested = fixture();
  const location = 'node_modules/tool/node_modules/braces';
  nested.packages[location] = {version: '3.0.2'};
  nested.report.vulnerabilities.braces.nodes.push(location);
  assert.match(decision(nested).failure, /braces/);
});

test('unrelated findings at every severity still block alongside accepted causes', () => {
  for (const severity of ['info', 'low', 'moderate', 'high', 'critical', 'constructor']) {
    const candidate = fixture();
    candidate.packages['node_modules/other'] = {version: '1.0.0'};
    candidate.report.vulnerabilities.other = finding('other', [{name: 'other', severity}], severity);
    candidate.report.vulnerabilities.tool.via.push('other');
    assert.ok(decision(candidate).failure, `${severity} advisory was hidden`);
    assert.equal(decision(candidate).accepted.has('root'), false);
  }
});

test('dangling edges, incomplete nodes, unknown severities and orphan cycles block', () => {
  for (const mutation of [
    (graph) => { graph.tool.via.push('absent'); },
    (graph) => { graph.tool.nodes = []; },
    (graph) => { graph.tool.nodes.push('node_modules/absent'); },
    (graph) => { graph.tool.name = 'different'; },
    (graph) => { graph.tool.severity = 'critical'; },
    (graph) => { graph.tool.severity = '__proto__'; },
    (graph) => { graph.tool.via = ['root']; },
    (graph) => { graph.root.via.push('orphan'); graph.orphan = finding('orphan', ['orphan']); },
  ]) {
    const candidate = fixture();
    candidate.packages['node_modules/orphan'] = {version: '1.0.0'};
    mutation(candidate.report.vulnerabilities);
    assert.ok(decision(candidate).failure);
    assert.equal(decision(candidate).accepted.has('root'), false);
  }
});

test('a complete cycle anchored in reviewed causes is order-independent', () => {
  const candidate = fixture();
  candidate.report.vulnerabilities.tool.via.push('root');
  assert.equal(decision(candidate).failure, null);
  candidate.report.vulnerabilities = Object.fromEntries(Object.entries(candidate.report.vulnerabilities).reverse());
  assert.equal(decision(candidate).failure, null);
});

test('invalid audit reports and unknown severity cannot use exception acceptance', () => {
  for (const report of [null, [], 'invalid', {error: {code: 'ENOAUDIT'}}, {}, {vulnerabilities: []}, {vulnerabilities: {braces: null}}]) {
    assert.equal(acceptedPackages(report, {}, review()).size, 0);
    assert.ok(auditFailure(report, new Map()));
  }
});

test('every installed copy must match the locked exception version and identity', (t) => {
  const site = mkdtempSync(path.join(tmpdir(), 'docs-exception-'));
  t.after(() => rmSync(site, {recursive: true, force: true}));
  const packages = {};
  const install = (location, name, version) => {
    const directory = path.join(site, location);
    mkdirSync(directory, {recursive: true});
    writeFileSync(path.join(directory, 'package.json'), JSON.stringify({name, version}));
    packages[location] = {version};
  };
  for (const entry of entries) install(`node_modules/${entry.package}`, entry.package, entry.version);
  assert.doesNotThrow(() => verifyInstalledExceptions(site, packages, review()));
  const nested = 'node_modules/tool/node_modules/braces';
  install(nested, 'braces', '3.0.2');
  assert.throws(() => verifyInstalledExceptions(site, packages, review()), /version mismatch/);
  packages[nested].version = '3.0.3';
  assert.throws(() => verifyInstalledExceptions(site, packages, review()), /version mismatch/);
  install(nested, 'renamed-braces', '3.0.3');
  assert.throws(() => verifyInstalledExceptions(site, packages, review()), /version mismatch/);
  delete packages[nested];
  delete packages['node_modules/braces'];
  assert.throws(() => verifyInstalledExceptions(site, packages, review()), /absent from lockfile/);
  packages['../node_modules/braces'] = {version: '3.0.3'};
  assert.throws(() => verifyInstalledExceptions(site, packages, review()), /Invalid locked package location/);
});

test('npm execution errors never become successful exception evidence', () => {
  const {report} = fixture();
  assert.equal(auditExecutionFailure({status: 0}, report, null), null);
  assert.equal(auditExecutionFailure({status: 1}, report, null), null);
  assert.equal(auditExecutionFailure({status: 1}, report, 'blocked'), 'blocked');
  assert.match(auditExecutionFailure({status: 1, error: new Error('spawn failed')}, report, null), /spawn failed/);
  assert.match(auditExecutionFailure({status: null, signal: 'SIGTERM'}, report, null), /SIGTERM/);
  for (const status of [2, 99, null]) assert.match(auditExecutionFailure({status}, report, null), /exited/);
  assert.match(auditExecutionFailure({status: 1}, {vulnerabilities: {}}, null), /without accounted/);
});
