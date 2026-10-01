import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

import {resolveRepositoryFile} from './repository-paths.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const siteRoot = path.resolve(path.dirname(scriptPath), '..');
const repoRoot = path.resolve(siteRoot, '..');
const docsMap = JSON.parse(
  await fs.readFile(path.resolve(siteRoot, 'docs-map.json'), 'utf8')
);

export function markdownSources(map) {
  return [
    ...new Set([
      ...map.documents.map((document) => document.source),
    ]),
  ].sort();
}

export function lintExitCode(status) {
  return status ?? 1;
}

async function main() {
  const frozen = new Map(docsMap.documents.filter((document) => document.frozenSha256).map((document) => [document.source, document.frozenSha256]));
  for (const [source, expected] of frozen) {
    const bytes = await fs.readFile(resolveRepositoryFile(repoRoot, source));
    const actual = createHash('sha256').update(bytes).digest('hex');
    if (actual !== expected) throw new Error(`frozen documentation source changed: ${source}`);
    process.stdout.write(`Frozen source matches adoption baseline: ${source}\n`);
  }
  const sources = markdownSources(docsMap).filter((source) => !frozen.has(source)).map(
    (source) => resolveRepositoryFile(repoRoot, source)
  );
  const result = spawnSync(
    'markdownlint-cli2',
    ['--config', '../.markdownlint.json', ...sources],
    {
      cwd: siteRoot,
      encoding: 'utf8',
      stdio: 'inherit',
    }
  );
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(lintExitCode(result.status));
  }
}

if (process.argv[1] === scriptPath) {
  await main();
}
