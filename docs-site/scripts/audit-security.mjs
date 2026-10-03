import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {acceptedPackages, auditExecutionFailure, reviewedExceptions, verifyInstalledExceptions} from './audit-exceptions.mjs';
import {auditFailure} from './audit-report.mjs';
import {productionRoots} from './audit-scope.mjs';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const result = spawnSync('npm', ['audit', '--omit=dev', '--json'], {
  cwd: siteRoot,
  encoding: 'utf8',
});
if (!result.stdout) {
  process.stderr.write(result.stderr || result.error?.message || 'npm audit produced no JSON output\n');
  process.exit(1);
}
const report = JSON.parse(result.stdout);
// Preserve npm's original findings in the retained command log, including exceptions.
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
const exceptions = JSON.parse(readFileSync(path.join(siteRoot, 'security-exceptions.json'), 'utf8'));
const reviewed = reviewedExceptions(exceptions);
const packages = JSON.parse(readFileSync(path.join(siteRoot, 'package-lock.json'), 'utf8')).packages;
verifyInstalledExceptions(siteRoot, packages, reviewed);
const accepted = acceptedPackages(report, packages, reviewed);
const failure = auditExecutionFailure(result, report, auditFailure(report, productionRoots(siteRoot), accepted));
if (failure) {
  process.stderr.write(`${failure}\n`);
  process.exit(1);
}
for (const entry of reviewed.values()) {
  if (accepted.has(entry.package)) {
    process.stdout.write(`Accepted until ${entry.expires} UTC: ${entry.id}, ${entry.package}@${entry.version}; ${entry.reason}\n`);
  }
}
process.stdout.write(`No unaccepted docs-site production advisories; ${accepted.size} reported package findings accepted\n`);
