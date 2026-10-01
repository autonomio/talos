import fs from 'node:fs';
import path from 'node:path';

export function isPathInside(root, candidate) {
  const relative = path.relative(root, candidate);
  return (
    relative === ''
    || (!relative.startsWith(`..${path.sep}`)
      && relative !== '..'
      && !path.isAbsolute(relative))
  );
}

export function resolveRepositoryPath(root, source) {
  const candidate = path.resolve(root, source);
  if (!isPathInside(root, candidate)) {
    throw new Error(`documentation source is outside the repository: ${source}`);
  }
  return candidate;
}

export function resolveRepositoryFile(root, source) {
  const candidate = resolveRepositoryPath(root, source);
  if (!fs.existsSync(candidate) || !fs.statSync(candidate).isFile()) {
    throw new Error(`documentation source is not a file: ${source}`);
  }
  const realRoot = fs.realpathSync(root);
  const realCandidate = fs.realpathSync(candidate);
  if (!isPathInside(realRoot, realCandidate)) {
    throw new Error(`documentation source resolves outside the repository: ${source}`);
  }
  return realCandidate;
}
