import {createHash} from 'node:crypto';
import {existsSync, lstatSync, readFileSync, realpathSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {installedLocations} from './audit-exceptions.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const siteRoot = path.resolve(path.dirname(scriptPath), '..');
const patchRoot = path.join(siteRoot, 'security-patches');
export const securityBackports = JSON.parse(readFileSync(path.join(patchRoot, 'manifest.json'), 'utf8'));
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

/** Apply exactly one identified unified file diff, refusing fuzz or source drift. */
export function applyPatch(before, patch, beforeSha256, afterSha256) {
  if (digest(before) !== beforeSha256) throw new Error('Security patch upstream source hash mismatch');
  const source = before.toString('utf8').match(/[^\n]*\n|[^\n]+$/g) ?? [];
  const lines = patch.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  if (lines.length < 3 || !lines[0].startsWith('--- ') || !lines[1].startsWith('+++ ')) {
    throw new Error('Security patch must contain one unified file diff');
  }
  const result = [];
  let cursor = 0;
  let index = 2;
  while (index < lines.length) {
    const match = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n$/.exec(lines[index]);
    if (!match) throw new Error('Security patch has an invalid hunk header');
    const start = Math.max(0, Number(match[1]) - 1);
    const oldCount = match[2] === undefined ? 1 : Number(match[2]);
    const newStart = Math.max(0, Number(match[3]) - 1);
    const newCount = match[4] === undefined ? 1 : Number(match[4]);
    if (start < cursor || start > source.length) throw new Error('Security patch hunks overlap or exceed source');
    result.push(...source.slice(cursor, start));
    cursor = start;
    if (result.length !== newStart) throw new Error('Security patch new hunk position mismatch');
    let consumed = 0;
    let emitted = 0;
    index++;
    while (index < lines.length && !lines[index].startsWith('@@ ')) {
      const operation = lines[index][0];
      const content = lines[index].slice(1);
      if (![' ', '+', '-'].includes(operation)) throw new Error('Security patch contains an unsupported operation');
      if (operation !== '+') {
        if (source[cursor] !== content) throw new Error('Security patch context mismatch');
        cursor++;
        consumed++;
      }
      if (operation !== '-') {
        result.push(content);
        emitted++;
      }
      index++;
    }
    if (consumed !== oldCount || emitted !== newCount) throw new Error('Security patch hunk size mismatch');
  }
  result.push(...source.slice(cursor));
  const after = Buffer.from(result.join(''), 'utf8');
  if (digest(after) !== afterSha256) throw new Error('Security patch result hash mismatch');
  return after;
}

/** Check all locked installed copies before returning any source mutations. */
export function patchPlan(root, entries) {
  const packages = JSON.parse(readFileSync(path.join(root, 'package-lock.json'), 'utf8')).packages;
  const installed = installedLocations(root, new Set(entries.map((entry) => entry.package)));
  const plan = [];
  for (const entry of entries) {
    const locations = [...installed].filter(([, name]) => name === entry.package).map(([location]) => location);
    const locked = Object.keys(packages).filter((location) => location.split('node_modules/').at(-1) === entry.package);
    if (locations.length === 0 || locked.some((location) => !locations.includes(location))) {
      throw new Error(`Security backport package is missing: ${entry.package}`);
    }
    for (const location of locations) {
      const base = path.join(root, location);
      const metadata = JSON.parse(readFileSync(path.join(base, 'package.json'), 'utf8'));
      if (metadata.name !== entry.package || metadata.version !== entry.version
          || packages[location]?.version !== entry.version) {
        throw new Error(`Security backport identity mismatch: ${location}`);
      }
      const nodeModules = realpathSync(path.join(root, 'node_modules'));
      if (!realpathSync(base).startsWith(nodeModules + path.sep)) {
        throw new Error(`Security backport cannot modify an external package directory: ${location}`);
      }
      for (const file of entry.files) {
        const target = path.join(base, file.path);
        if (existsSync(target) && lstatSync(target).isSymbolicLink()) {
          throw new Error(`Security backport source cannot be a symlink: ${target}`);
        }
        const realBase = realpathSync(base);
        const realParent = realpathSync(path.dirname(target));
        if (realParent !== realBase && !realParent.startsWith(realBase + path.sep)) {
          throw new Error(`Security backport source directory escapes its package: ${target}`);
        }
        const before = existsSync(target) ? readFileSync(target) : Buffer.alloc(0);
        const alreadyPatched = digest(before) === file.after_sha256;
        const after = alreadyPatched ? before : applyPatch(before,
          readFileSync(path.join(patchRoot, file.patch), 'utf8'), file.before_sha256, file.after_sha256);
        plan.push({target, before, after, alreadyPatched, advisory: entry.advisory});
      }
    }
  }
  return plan;
}

/** Refuse advisory acceptance unless every installed source has the repaired identity. */
export function verifySecurityPatches(root) {
  for (const item of patchPlan(root, securityBackports)) {
    if (!item.alreadyPatched) throw new Error(`Security backport missing for ${item.advisory}: ${item.target}`);
  }
}

if (process.argv[1] === scriptPath) {
  const plan = patchPlan(siteRoot, securityBackports);
  for (const item of plan) if (!item.alreadyPatched) writeFileSync(item.target, item.after);
  verifySecurityPatches(siteRoot);
  process.stdout.write(`Verified ${plan.length} installed source files for ${securityBackports.length} explicit security backports\n`);
}
