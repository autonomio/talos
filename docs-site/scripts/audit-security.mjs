import {spawnSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {auditFailure} from './audit-report.mjs';
import {productionRoots} from './audit-scope.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const siteRoot = path.resolve(path.dirname(scriptPath), '..');
const result = spawnSync('npm', ['audit', '--omit=dev', '--json'], {
  cwd: siteRoot,
  encoding: 'utf8',
});
if (!result.stdout) {
  process.stderr.write(result.stderr || 'npm audit produced no JSON output\n');
  process.exit(result.status || 1);
}
const report = JSON.parse(result.stdout);
const failure = auditFailure(report, productionRoots(siteRoot));
if (failure) {
  process.stderr.write(`${failure}\n`);
  process.exit(1);
}
process.stdout.write('No docs-site npm vulnerabilities found\n');
