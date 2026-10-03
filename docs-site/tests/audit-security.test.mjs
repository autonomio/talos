import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {copyFileSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import test from 'node:test';

function fixture(t) {
  const root = mkdtempSync(path.join(tmpdir(), 'docs-audit-cli-'));
  t.after(() => rmSync(root, {recursive: true, force: true}));
  mkdirSync(path.join(root, 'scripts'));
  mkdirSync(path.join(root, 'bin'));
  for (const name of ['audit-security.mjs', 'audit-report.mjs', 'audit-exceptions.mjs', 'audit-scope.mjs']) {
    copyFileSync(new URL(`../scripts/${name}`, import.meta.url), path.join(root, 'scripts', name));
  }
  const entries = JSON.parse(readFileSync(new URL('../security-exceptions.json', import.meta.url), 'utf8'));
  const now = Date.now();
  for (const entry of entries) {
    entry.approved_on = new Date(now).toISOString().slice(0, 10);
    entry.expires = new Date(now + 86400000).toISOString().slice(0, 10);
  }
  writeFileSync(path.join(root, 'security-exceptions.json'), JSON.stringify(entries));
  const dependencies = Object.fromEntries(entries.map((entry) => [entry.package, entry.version]));
  const packages = {'node_modules/root': {version: '1.0.0', dependencies}};
  const vulnerabilities = {root: {name: 'root', severity: 'high', via: Object.keys(dependencies), nodes: ['node_modules/root']}};
  for (const entry of entries) {
    const location = `node_modules/${entry.package}`;
    packages[location] = {version: entry.version};
    mkdirSync(path.join(root, location), {recursive: true});
    writeFileSync(path.join(root, location, 'package.json'), JSON.stringify({name: entry.package, version: entry.version}));
    vulnerabilities[entry.package] = {name: entry.package, severity: entry.severity, nodes: [location], via: [{
      name: entry.package, dependency: entry.package, severity: entry.severity,
      url: `https://github.com/advisories/${entry.id}`,
    }]};
  }
  writeFileSync(path.join(root, 'package.json'), JSON.stringify({dependencies: {root: '1.0.0'}}));
  writeFileSync(path.join(root, 'package-lock.json'), JSON.stringify({packages}));
  const report = {auditReportVersion: 2, vulnerabilities};
  const run = (payload = report, status = 1, signal = false) => {
    const script = `#!/usr/bin/env node
if (JSON.stringify(process.argv.slice(2)) !== '["audit","--omit=dev","--json"]') process.exit(98);
process.stdout.write(${JSON.stringify(JSON.stringify(payload))}, () => {
${signal ? "process.kill(process.pid, 'SIGTERM');" : `process.exit(${status});`}
});
`;
    writeFileSync(path.join(root, 'bin/npm'), script, {mode: 0o755});
    return spawnSync(process.execPath, [path.join(root, 'scripts/audit-security.mjs')], {
      env: {...process.env, PATH: `${root}/bin:${process.env.PATH}`}, encoding: 'utf8',
    });
  };
  return {root, entries, report, run};
}

test('the audit CLI retains original findings and explicitly records accepted npm exit 1', (t) => {
  const {run, report, entries} = fixture(t);
  const result = run();
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stdout.startsWith(JSON.stringify(report, null, 2)));
  for (const entry of entries) assert.ok(result.stdout.includes(`${entry.id}, ${entry.package}@${entry.version}`));
  assert.match(result.stdout, /3 reported package findings accepted/);
});

test('the audit CLI rejects unrelated findings, auditor errors, signals and empty error exits', (t) => {
  const {run, report} = fixture(t);
  const extra = structuredClone(report);
  extra.vulnerabilities.braces.via.push({...extra.vulnerabilities.braces.via[0], url: 'https://github.com/advisories/GHSA-abcd-abcd-abcd'});
  for (const [payload, status, signal] of [
    [extra, 1, false],
    [{error: {code: 'ENOAUDIT'}}, 1, false],
    [{vulnerabilities: {}}, 1, false],
    [report, 2, false],
    [report, 99, false],
    [report, 0, true],
    [{vulnerabilities: []}, 0, false],
  ]) {
    const result = run(payload, status, signal);
    assert.equal(result.status, 1, JSON.stringify({stdout: result.stdout, stderr: result.stderr}));
    assert.ok(result.stdout.startsWith(JSON.stringify(payload, null, 2)));
    assert.doesNotMatch(result.stdout, /No unaccepted/);
  }
});

test('the audit CLI rejects expired approval and installed-package tampering', (t) => {
  const {root, run, entries} = fixture(t);
  const metadata = path.join(root, 'node_modules/braces/package.json');
  writeFileSync(metadata, JSON.stringify({name: 'braces', version: '3.0.2'}));
  const changed = run();
  assert.equal(changed.status, 1);
  assert.match(changed.stderr, /version mismatch/);
  writeFileSync(metadata, JSON.stringify({name: 'braces', version: '3.0.3'}));
  for (const entry of entries) {
    entry.approved_on = new Date(Date.now() - 5 * 86400000).toISOString().slice(0, 10);
    entry.expires = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  }
  writeFileSync(path.join(root, 'security-exceptions.json'), JSON.stringify(entries));
  const expired = run();
  assert.equal(expired.status, 1);
  assert.match(expired.stderr, /expired/);
  assert.ok(expired.stdout.includes(entries[0].id), 'original advisory disappeared');
});
