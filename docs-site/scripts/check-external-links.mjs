import dns from 'node:dns/promises';
import fs from 'node:fs/promises';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import MarkdownIt from 'markdown-it';

import {resolveRepositoryFile} from './repository-paths.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), '..', '..');
const docsMap = JSON.parse(
  await fs.readFile(path.resolve(repoRoot, 'docs-site', 'docs-map.json'), 'utf8')
);

function stripInlineCode(line) {
  let output = '';
  let index = 0;
  while (index < line.length) {
    const opening = line.indexOf('`', index);
    if (opening === -1) {
      return output + line.slice(index);
    }
    output += line.slice(index, opening);
    let markerEnd = opening;
    while (line[markerEnd] === '`') {
      markerEnd += 1;
    }
    const marker = line.slice(opening, markerEnd);
    const closing = line.indexOf(marker, markerEnd);
    if (closing === -1) {
      return output;
    }
    index = closing + marker.length;
  }
  return output;
}

export function stripMarkdownCode(text) {
  let fence = null;
  return text.split('\n').map((line) => {
    const match = line.match(/^\s*(`{3,}|~{3,})/);
    if (match) {
      const marker = match[1];
      if (fence === null) {
        fence = marker;
      } else if (marker[0] === fence[0] && marker.length >= fence.length) {
        fence = null;
      }
      return '';
    }
    return fence === null ? stripInlineCode(line) : '';
  }).join('\n');
}

const markdownParser = new MarkdownIt({html: true});

export function extractExternalLinks(text) {
  const links = new Set();
  function add(value) {
    if (/^https?:\/\//.test(value || '')) links.add(value);
  }
  function visit(tokens) {
    for (const token of tokens) {
      if (token.type === 'link_open') add(token.attrGet('href'));
      if (token.type === 'image') add(token.attrGet('src'));
      if (token.type === 'html_inline' || token.type === 'html_block') {
        for (const match of token.content.matchAll(/(?:href|src)=["'](https?:\/\/[^"']+)["']/g)) add(match[1]);
      }
      if (token.children) visit(token.children);
    }
  }
  visit(markdownParser.parse(text, {}));
  return links;
}

function isPrivateIpv4(address) {
  const [first, second, third] = address.split('.').map(Number);
  return (
    first === 0
    || first === 10
    || first === 127
    || (first === 100 && second >= 64 && second <= 127)
    || (first === 169 && second === 254)
    || (first === 172 && second >= 16 && second <= 31)
    || (first === 192 && second === 0 && [0, 2].includes(third))
    || (first === 192 && second === 168)
    || (first === 198 && (second === 18 || second === 19))
    || address.startsWith('198.51.100.')
    || address.startsWith('203.0.113.')
    || first >= 224
  );
}

export function isPublicAddress(address) {
  const normalized = address.toLowerCase();
  if (net.isIP(normalized) === 4) {
    return !isPrivateIpv4(normalized);
  }
  if (net.isIP(normalized) !== 6) {
    return false;
  }
  if (normalized.startsWith('::ffff:')) {
    return isPublicAddress(normalized.slice('::ffff:'.length));
  }
  return !(
    normalized === '::'
    || normalized === '::1'
    || normalized.startsWith('fc')
    || normalized.startsWith('fd')
    || /^fe[89ab]/.test(normalized)
    || normalized.startsWith('ff')
    || normalized.startsWith('2001:db8:')
  );
}

export async function assertPublicUrl(url) {
  const parsed = new URL(url);
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`${url} must use HTTP or HTTPS`);
  }
  if (parsed.username || parsed.password) {
    throw new Error(`${url} must not contain credentials`);
  }
  const hostname = parsed.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (hostname === 'localhost' || hostname.endsWith('.localhost')) {
    throw new Error(`${url} resolves to a non-public destination`);
  }
  const family = net.isIP(hostname);
  const addresses = family
    ? [{address: hostname, family}]
    : await dns.lookup(hostname, {all: true});
  if (
    addresses.length === 0
    || addresses.some(({address}) => !isPublicAddress(address))
  ) {
    throw new Error(`${url} resolves to a non-public destination`);
  }
  return {addresses, parsed};
}

function pinnedLookup(addresses) {
  return (_hostname, options, callback) => {
    if (options.all) {
      callback(null, addresses);
      return;
    }
    const [{address, family}] = addresses;
    callback(null, address, family);
  };
}

async function requestOnce(url, method) {
  const {addresses, parsed} = await assertPublicUrl(url);
  const transport = parsed.protocol === 'https:' ? https : http;
  return new Promise((resolve, reject) => {
    const outgoing = transport.request(
      parsed,
      {
        headers: {'user-agent': 'autonomio-docs-link-check/1.0'},
        lookup: pinnedLookup(addresses),
        method,
        signal: AbortSignal.timeout(15000),
      },
      resolve
    );
    outgoing.on('error', reject);
    outgoing.end();
  });
}

async function request(url, method) {
  let currentUrl = url;
  for (let redirectCount = 0; redirectCount <= 5; redirectCount += 1) {
    const response = await requestOnce(currentUrl, method);
    if (![301, 302, 303, 307, 308].includes(response.statusCode)) {
      return response;
    }
    const location = response.headers.location;
    response.resume();
    if (!location) {
      throw new Error(`${currentUrl} redirected without a location`);
    }
    currentUrl = new URL(location, currentUrl).href;
  }
  throw new Error(`${url} exceeded five redirects`);
}

// Statuses that mean "not now" rather than "this link is wrong". Retrying the
// second class -- 404 and its neighbours -- would turn a real broken link into
// a slow real broken link, which is the opposite of what this check is for.
export const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);
// Bounded on purpose: an unbounded retry converts a dead host into a hung
// required check, which is worse than a red one.
export const MAX_ATTEMPTS = 3;
const RETRY_BASE_MS = 500;
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export async function attemptLink(url, requestFunction = request) {
  const response = await requestFunction(url, 'HEAD');
  // Some public sites reject HEAD or route it differently. A successful GET
  // proves the reader's actual access path; a failed GET still blocks.
  if ([400, 403, 404, 405, 410, 501].includes(response.statusCode)) {
    response.resume();
    const getResponse = await requestFunction(url, 'GET');
    getResponse.resume();
    return getResponse.statusCode;
  }
  response.resume();
  return response.statusCode;
}

export async function checkLink(url, attempt = async (u) => attemptLink(u), sleep = delay) {
  let lastStatus = null;
  for (let tries = 1; tries <= MAX_ATTEMPTS; tries += 1) {
    let status;
    try {
      status = await attempt(url);
    } catch (error) {
      // A network-level failure -- DNS, reset, the 15s timeout -- is the same
      // class as a 5xx: the host, not the link.
      if (tries === MAX_ATTEMPTS) {
        throw error;
      }
      await sleep(RETRY_BASE_MS * tries);
      continue;
    }
    if (status >= 200 && status < 300) {
      return;
    }
    lastStatus = status;
    if (!RETRYABLE_STATUS.has(status) || tries === MAX_ATTEMPTS) {
      break;
    }
    await sleep(RETRY_BASE_MS * tries);
  }
  throw new Error(`${url} returned ${lastStatus}`);
}

export function localSiteSource(url, profile, documents, sections = []) {
  const target = new URL(url);
  if (target.origin !== new URL(profile.siteUrl).origin || !target.pathname.startsWith(profile.basePath)) return null;
  const route = '/' + decodeURIComponent(target.pathname.slice(profile.basePath.length)).replace(/\/$/, '');
  if (route.split('/').includes('..')) throw new Error('local site URL escapes its static directory');
  const document = documents.find((item) => item.slug === route);
  if (document) return document.source;
  if (sections.some((section) => section.slug === route)) return 'docs-site/product-docs.json';
  return 'docs-site/static' + route;
}

async function main() {
  const links = new Set();
  for (const document of docsMap.documents) {
    const sourcePath = resolveRepositoryFile(repoRoot, document.source);
    const text = await fs.readFile(sourcePath, 'utf8');
    for (const url of extractExternalLinks(text)) {
      links.add(url);
    }
  }
  const profile = JSON.parse(await fs.readFile(path.resolve(repoRoot, 'docs-site/product-docs.json'), 'utf8'));
  for (const url of [...links].sort()) {
    const source = localSiteSource(url, profile, docsMap.documents, docsMap.sections);
    if (source === null) await checkLink(url);
    else await fs.access(resolveRepositoryFile(repoRoot, source));
  }
  process.stdout.write(`External links healthy: ${links.size}\n`);
}

if (process.argv[1] === scriptPath) {
  await main();
}
