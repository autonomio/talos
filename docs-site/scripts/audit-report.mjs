// Every known production advisory blocks documentation acceptance.
// Patched dependency overrides are committed in package.json and its lockfile.
export const RELAXED_ROOTS = Object.freeze([]);
export const RELAXED_FLOOR = 'info';
export const DEFAULT_FLOOR = 'info';

const RANK = new Map([
  ['info', 0],
  ['low', 1],
  ['moderate', 2],
  ['high', 3],
  ['critical', 4],
]);

function floorFor(roots) {
  const relaxed = roots !== undefined
    && roots.size > 0
    && [...roots].every((root) => RELAXED_ROOTS.includes(root));
  return relaxed ? RELAXED_FLOOR : DEFAULT_FLOOR;
}

/**
 * Describe why the audit blocks, or return null when it does not.
 *
 * `rootsByPackage` maps a package name to the direct dependencies that reach
 * it, as produced by `audit-scope.mjs`.
 */
export function auditFailure(report, rootsByPackage) {
  if (Object.hasOwn(report, 'error')) {
    return `npm audit failed: ${JSON.stringify(report.error)}`;
  }
  if (
    !Object.hasOwn(report, 'vulnerabilities')
    || typeof report.vulnerabilities !== 'object'
    || report.vulnerabilities === null
  ) {
    return 'npm audit report has no vulnerabilities object';
  }

  const blocking = [];
  for (const [name, vulnerability] of Object.entries(report.vulnerabilities)) {
    const rank = RANK.get(vulnerability.severity);
    if (rank === undefined) {
      return `npm audit reported unknown severity ${JSON.stringify(vulnerability.severity)} `
        + `for ${name}`;
    }
    const roots = rootsByPackage.get(name);
    const floor = floorFor(roots);
    if (rank >= RANK.get(floor)) {
      const via = roots === undefined ? 'unresolved' : [...roots].sort().join(', ');
      blocking.push(`${name} (${vulnerability.severity}, floor ${floor}, via ${via})`);
    }
  }

  return blocking.length > 0
    ? `Docs-site npm vulnerabilities at or above their severity floor:\n  ${blocking.sort().join('\n  ')}`
    : null;
}
