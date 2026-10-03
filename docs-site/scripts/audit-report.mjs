// Every production advisory blocks unless its complete cause graph has a reviewed exception.
export const DEFAULT_FLOOR = 'info';

// A Map, not an object literal: `RANK.constructor` and `RANK.__proto__` resolve
// through `Object.prototype` on a literal, so a severity string colliding with
// an inherited key would pass the unknown-severity guard below as a function and
// then compare false against the floor -- ranking the advisory as harmless,
// which is the behaviour that guard exists to prevent.
const RANK = new Map([
  ['info', 0],
  ['low', 1],
  ['moderate', 2],
  ['high', 3],
  ['critical', 4],
]);

/**
 * Describe why the audit blocks, or return null when it does not.
 *
 * `rootsByPackage` maps a package name to the direct dependencies that reach
 * it, as produced by `audit-scope.mjs`.
 */
export function auditFailure(report, rootsByPackage, accepted = new Set()) {
  if (typeof report !== 'object' || report === null || Array.isArray(report)) {
    return 'npm audit report is not an object';
  }
  if (Object.hasOwn(report, 'error')) {
    return `npm audit failed: ${JSON.stringify(report.error)}`;
  }
  if (
    !Object.hasOwn(report, 'vulnerabilities')
    || typeof report.vulnerabilities !== 'object'
    || report.vulnerabilities === null
    || Array.isArray(report.vulnerabilities)
  ) {
    return 'npm audit report has no vulnerabilities object';
  }

  const blocking = [];
  for (const [name, vulnerability] of Object.entries(report.vulnerabilities)) {
    if (typeof vulnerability !== 'object' || vulnerability === null || Array.isArray(vulnerability)) {
      return `npm audit reported an invalid vulnerability for ${name}`;
    }
    const rank = RANK.get(vulnerability.severity);
    if (rank === undefined) {
      return `npm audit reported unknown severity ${JSON.stringify(vulnerability.severity)} `
        + `for ${name}`;
    }
    const roots = rootsByPackage.get(name);
    const floor = DEFAULT_FLOOR;
    const reviewed = accepted.has(name) && roots !== undefined && roots.size > 0;
    if (rank >= RANK.get(floor) && !reviewed) {
      const via = roots === undefined ? 'unresolved' : [...roots].sort().join(', ');
      blocking.push(`${name} (${vulnerability.severity}, floor ${floor}, via ${via})`);
    }
  }

  return blocking.length > 0
    ? `Docs-site production advisories:\n  ${blocking.sort().join('\n  ')}`
    : null;
}
