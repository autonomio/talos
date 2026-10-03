import {readFileSync} from 'node:fs';
import path from 'node:path';

const DAY = 24 * 60 * 60 * 1000;
const SEVERITIES = Object.freeze(['info', 'low', 'moderate', 'high', 'critical']);
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value) => typeof value === 'string' && value.trim().length > 0;
const packageName = (location) => location.split('node_modules/').at(-1);

function date(value) {
  const timestamp = Date.parse(`${value}T00:00:00Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || !Number.isFinite(timestamp)
      || new Date(timestamp).toISOString().slice(0, 10) !== value) {
    throw new Error(`Invalid documentation exception date: ${value}`);
  }
  return timestamp;
}

/** Validate reviewed, exact-version exceptions and their exclusive UTC expiry. */
export function reviewedExceptions(entries, now = new Date()) {
  if (!Array.isArray(entries) || !Number.isFinite(now.getTime())) {
    throw new Error('Invalid documentation exceptions or audit time');
  }
  const reviewed = new Map();
  for (const entry of entries) {
    if (!object(entry) || !/^GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}$/.test(entry.id)
        || !text(entry.package) || !/^\d+\.\d+\.\d+(?:-[a-z0-9.-]+)?$/.test(entry.version)
        || !SEVERITIES.includes(entry.severity)
        || !text(entry.reason) || !text(entry.approved_by)) {
      throw new Error('Documentation exception requires an advisory ID, exact package/version, severity, owner and reason');
    }
    const approved = date(entry.approved_on);
    const expires = date(entry.expires);
    if (expires <= approved || expires - approved > 30 * DAY || approved > now.getTime()) {
      throw new Error(`Invalid review window for ${entry.id}`);
    }
    if (now.getTime() >= expires) {
      throw new Error(`Documentation exception expired: ${entry.id} (${entry.expires})`);
    }
    if (reviewed.has(entry.id)) {
      throw new Error(`Duplicate documentation exception: ${entry.id}`);
    }
    reviewed.set(entry.id, entry);
  }
  return reviewed;
}

/** Bind every installed copy of an excepted package to its exact lockfile version. */
export function verifyInstalledExceptions(siteRoot, packages, reviewed) {
  for (const entry of reviewed.values()) {
    const locations = Object.keys(packages).filter((location) => packageName(location) === entry.package);
    if (locations.length === 0) {
      throw new Error(`Excepted package is absent from lockfile: ${entry.package}; remove its exception`);
    }
    for (const location of locations) {
      if (!location.startsWith('node_modules/') || location.split('/').some((part) => ['.', '..', ''].includes(part))) {
        throw new Error(`Invalid locked package location: ${location}`);
      }
      const installed = JSON.parse(readFileSync(path.join(siteRoot, location, 'package.json'), 'utf8'));
      if (packages[location].version !== entry.version || installed.version !== entry.version
          || installed.name !== entry.package) {
        throw new Error(`Documentation exception version mismatch: ${entry.package} at ${location}`);
      }
    }
  }
}

function acceptedAdvisory(advisory, name, vulnerability, packages, reviewed) {
  if (!object(advisory) || typeof advisory.url !== 'string') return false;
  const id = advisory.url.replace('https://github.com/advisories/', '');
  const entry = reviewed.get(id);
  return entry !== undefined && advisory.url === `https://github.com/advisories/${entry.id}`
    && entry.package === name && advisory.name === name && advisory.dependency === name
    && advisory.severity === entry.severity && vulnerability.severity === entry.severity
    && Array.isArray(vulnerability.nodes) && vulnerability.nodes.length > 0
    && vulnerability.nodes.every((location) => Object.hasOwn(packages, location)
      && packageName(location) === name && packages[location].version === entry.version);
}

/** Accept a propagated finding only if its complete cause graph is reviewed. */
export function acceptedPackages(report, packages, reviewed) {
  const accepted = new Set();
  if (!object(report) || !object(report.vulnerabilities) || Object.hasOwn(report, 'error')) return accepted;
  const graph = report.vulnerabilities;
  const entries = Object.entries(graph);
  // A cycle must reach an advisory; a cyclic-only component is incomplete proof.
  const reachesAdvisory = new Set(entries.filter(([, value]) => object(value)
    && Array.isArray(value.via) && value.via.some(object)).map(([name]) => name));
  let previousSize;
  do {
    previousSize = reachesAdvisory.size;
    for (const [name, value] of entries) {
      if (object(value) && Array.isArray(value.via)
          && value.via.some((cause) => typeof cause === 'string' && reachesAdvisory.has(cause))) {
        reachesAdvisory.add(name);
      }
    }
  } while (previousSize !== reachesAdvisory.size);

  for (const [name] of entries) {
    const pending = [name];
    const seen = new Set();
    let complete = true;
    let maximumReported = -1;
    let maximumReviewed = -1;
    while (pending.length > 0 && complete) {
      const current = pending.pop();
      if (seen.has(current)) continue;
      seen.add(current);
      const value = Object.hasOwn(graph, current) ? graph[current] : null;
      if (!object(value) || value.name !== current || !reachesAdvisory.has(current)
          || !SEVERITIES.includes(value.severity)
          || !Array.isArray(value.nodes) || value.nodes.length === 0
          || !value.nodes.every((location) => Object.hasOwn(packages, location)
            && packageName(location) === current && text(packages[location].version))
          || !Array.isArray(value.via) || value.via.length === 0) {
        complete = false;
        break;
      }
      maximumReported = Math.max(maximumReported, SEVERITIES.indexOf(value.severity));
      for (const cause of value.via) {
        if (typeof cause === 'string') pending.push(cause);
        else if (!acceptedAdvisory(cause, current, value, packages, reviewed)) complete = false;
        else maximumReviewed = Math.max(maximumReviewed, SEVERITIES.indexOf(cause.severity));
      }
    }
    if (complete && maximumReported <= maximumReviewed) accepted.add(name);
  }
  return accepted;
}

/** npm exit 1 is admissible only when its nonempty findings are all accounted for. */
export function auditExecutionFailure(result, report, failure) {
  if (result.error) return `npm audit failed: ${result.error.message}`;
  if (result.signal) return `npm audit terminated by ${result.signal}`;
  if (failure) return failure;
  if (![0, 1].includes(result.status)
      || (result.status === 1 && Object.keys(report.vulnerabilities).length === 0)) {
    return `npm audit exited ${result.status} without accounted advisory findings`;
  }
  return null;
}
