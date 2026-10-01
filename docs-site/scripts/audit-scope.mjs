import {readFileSync} from 'node:fs';
import path from 'node:path';

const NODE_MODULES = 'node_modules/';

// npm resolves a dependency by walking up the `node_modules` chain from the
// requiring package to the tree root, taking the first match. Reproduce that
// walk rather than assuming a flat, hoisted tree: docusaurus nests several
// duplicate versions, and a flat lookup silently attributes them to the wrong
// parent.
function resolveFrom(packages, requirePath, name) {
  let base = requirePath;
  for (;;) {
    const candidate = base === '' ? NODE_MODULES + name : `${base}/${NODE_MODULES}${name}`;
    if (packages[candidate] !== undefined) {
      return candidate;
    }
    if (base === '') {
      return null;
    }
    const cut = base.lastIndexOf('/' + NODE_MODULES);
    base = cut === -1 ? '' : base.slice(0, cut);
  }
}

function packageName(packages, packagePath) {
  return packages[packagePath].name
    ?? packagePath.slice(packagePath.lastIndexOf(NODE_MODULES) + NODE_MODULES.length);
}

// npm installs and audits optional and peer edges too. Walking only
// `dependencies` would attribute a package that a strict root reaches solely
// through one of those edges to the docusaurus roots alone, which is exactly
// how a high-severity advisory would stop blocking. Adding edges can only widen
// a package's root set, and a wider set can only make `floorFor` stricter.
const EDGE_KINDS = Object.freeze(['dependencies', 'optionalDependencies', 'peerDependencies']);

function edgeNames(entry) {
  const names = new Set();
  for (const kind of EDGE_KINDS) {
    for (const name of Object.keys(entry[kind] ?? {})) {
      names.add(name);
    }
  }
  return names;
}

/**
 * Map each package in the tree to the set of direct dependencies that reach it.
 *
 * A package reachable only through `@docusaurus/core` maps to that one root; a
 * package shared with `react` maps to both. The severity floor in
 * `audit-report.mjs` is chosen from this set, so a shared dependency cannot
 * inherit a relaxed floor from one of its parents alone.
 */
export function rootsFrom(packages, directDependencies) {
  const roots = new Map();
  for (const root of directDependencies) {
    const start = resolveFrom(packages, '', root);
    if (start === null) {
      // Dropping the root silently would relax, not tighten: every package it
      // shares with the docusaurus stack would be left with docusaurus-only
      // attribution and fall to the `critical` floor. A lockfile that cannot
      // place a declared production dependency is out of sync, and the gate
      // says so instead of guessing.
      throw new Error(
        `production dependency '${root}' is declared in package.json but has no entry `
        + 'in package-lock.json. Run `npm install` to resync the lockfile: severity '
        + 'floors cannot be attributed while a production root is unplaceable.'
      );
    }
    const pending = [start];
    const visited = new Set([start]);
    while (pending.length > 0) {
      const current = pending.pop();
      const name = packageName(packages, current);
      if (!roots.has(name)) {
        roots.set(name, new Set());
      }
      roots.get(name).add(root);
      for (const dependency of edgeNames(packages[current])) {
        // An unresolvable transitive edge is ordinary: optional dependencies go
        // uninstalled and peers are satisfied by the parent. Only a declared
        // production root is required to resolve.
        const resolved = resolveFrom(packages, current, dependency);
        if (resolved !== null && !visited.has(resolved)) {
          visited.add(resolved);
          pending.push(resolved);
        }
      }
    }
  }
  return roots;
}

/**
 * Reachability for the site's production tree, read from `package-lock.json`.
 *
 * The lockfile is checked in and fully resolved, so this needs no install step
 * and returns the same answer on every machine.
 */
export function productionRoots(siteRoot) {
  const lock = JSON.parse(readFileSync(path.join(siteRoot, 'package-lock.json'), 'utf8'));
  const manifest = JSON.parse(readFileSync(path.join(siteRoot, 'package.json'), 'utf8'));
  return rootsFrom(lock.packages, Object.keys(manifest.dependencies ?? {}));
}
