import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {
  isPathInside,
  resolveRepositoryFile,
} from './repository-paths.mjs';
import {canonicalSitemapUrl, siteRoute} from './site-urls.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), '..', '..');
const siteRoot = path.resolve(repoRoot, 'docs-site');
const outRoot = path.resolve(siteRoot, '.generated', 'docs');
const staticRoot = path.resolve(siteRoot, '.generated', 'static');
const profile = JSON.parse(
  await fs.readFile(path.resolve(siteRoot, 'product-docs.json'), 'utf8')
);
const docsMap = JSON.parse(
  await fs.readFile(path.resolve(siteRoot, 'docs-map.json'), 'utf8')
);
const documents = docsMap.documents;
const sections = docsMap.sections;

function normalizePath(value) {
  return value.split(path.sep).join('/');
}

function requireString(object, key, context) {
  const value = object[key];
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${context}.${key} must be a non-empty string`);
  }
  return value;
}

function assertUnique(values, label) {
  if (new Set(values).size !== values.length) {
    throw new Error(`${label} values must be unique`);
  }
}

function requireRouteSlug(object, context) {
  const value = requireString(object, 'slug', context);
  const segments = value.split('/').slice(1);
  if (
    value !== '/'
    && (
      !value.startsWith('/')
      || value.endsWith('/')
      || segments.some(
        (segment) => !segment || segment === '.' || segment === '..' || /[?#]/.test(segment)
      )
    )
  ) {
    throw new Error(
      `${context}.slug must be / or a canonical leading-slash route without a trailing slash`
    );
  }
  return value;
}

export function validateSections(sectionValues) {
  if (!Array.isArray(sectionValues) || sectionValues.length !== 5) {
    throw new Error('docs-map.json must define the five standard sections');
  }
  for (const section of sectionValues) {
    requireString(section, 'dir', 'section');
    requireString(section, 'label', 'section');
    requireRouteSlug(section, 'section');
    requireString(section, 'description', 'section');
    if (!Number.isInteger(section.position) || section.position < 1) {
      throw new Error('section.position must be a positive integer');
    }
  }
  assertUnique(sectionValues.map((section) => section.dir), 'section dir');
  assertUnique(sectionValues.map((section) => section.slug), 'section slug');
  assertUnique(sectionValues.map((section) => section.position), 'section position');
}

export function validateDocuments(documentValues) {
  if (!Array.isArray(documentValues)) {
    throw new Error('docs-map.json documents must be an array');
  }
  for (const document of documentValues) {
    requireString(document, 'source', 'document');
    requireString(document, 'dest', 'document');
    requireRouteSlug(document, 'document');
  }
  assertUnique(documentValues.map((document) => document.source), 'document source');
  assertUnique(documentValues.map((document) => document.dest), 'document dest');
  assertUnique(documentValues.map((document) => document.slug), 'document slug');
  assertUnique(documentValues.flatMap((document) => document.legacySlugs || []), 'legacy slug');
}

export function validateProfile(profileValue) {
  for (const key of [
    'productId',
    'productName',
    'tagline',
    'siteUrl',
    'basePath',
    'sourceRepoUrl',
  ]) {
    requireString(profileValue, key, 'product-docs.json');
  }
  const siteUrl = profileValue.siteUrl;
  if (
    !URL.canParse(siteUrl)
    || !['http:', 'https:'].includes(new URL(siteUrl).protocol)
    || new URL(siteUrl).origin !== siteUrl
  ) {
    throw new Error(
      'product-docs.json.siteUrl must be an HTTP(S) origin without a trailing slash'
    );
  }
  const basePath = profileValue.basePath;
  if (basePath !== '/' && !/^\/[^/]+(?:\/[^/]+)*\/$/.test(basePath)) {
    throw new Error(
      'product-docs.json.basePath must be / or have leading and trailing slashes'
    );
  }
}

export function maintainedMarkdownSources(root = repoRoot) {
  const sources = fsSync.readdirSync(root, {withFileTypes: true})
    .filter((entry) => entry.isFile() && entry.name.endsWith('.md'))
    .map((entry) => entry.name);
  sources.push('talos/README.md', 'docs-site/README.md');
  function collect(directory) {
    for (const entry of fsSync.readdirSync(directory, {withFileTypes: true})) {
      const entryPath = path.resolve(directory, entry.name);
      if (entry.isDirectory()) collect(entryPath);
      else if (entry.name.endsWith('.md') && !entry.name.startsWith('_')) {
        sources.push(normalizePath(path.relative(root, entryPath)));
      }
    }
  }
  collect(path.resolve(root, 'docs'));
  return sources.sort();
}

export function validateInventory(documentValues, sourceValues) {
  const mapped = new Set(documentValues.map((document) => document.source));
  const expected = new Set(sourceValues);
  const missing = [...expected].filter((source) => !mapped.has(source));
  const unexpected = [...mapped].filter((source) => !expected.has(source));
  if (missing.length || unexpected.length) {
    throw new Error(`maintained Markdown inventory differs: missing=${missing.join(',')}; unexpected=${unexpected.join(',')}`);
  }
}

function validateConfiguration() {
  validateProfile(profile);
  validateSections(sections);
  validateDocuments(documents);
  validateInventory(documents, maintainedMarkdownSources());

  for (const doc of documents) {
    resolveRepositoryFile(repoRoot, doc.source);
    const destPath = path.resolve(outRoot, doc.dest);
    if (!destPath.startsWith(`${outRoot}${path.sep}`)) {
      throw new Error(`document destination is outside generated docs: ${doc.dest}`);
    }
  }
}

validateConfiguration();
const sourceBranch = profile.sourceBranch || 'main';
const repoBlobBaseUrl = `${profile.sourceRepoUrl}/blob/${sourceBranch}`;
const repoEditBaseUrl = `${profile.sourceRepoUrl}/edit/${sourceBranch}`;
const repoTreeBaseUrl = `${profile.sourceRepoUrl}/tree/${sourceBranch}`;
const mappingBySource = new Map(
  documents.map((doc) => [normalizePath(doc.source), doc])
);

async function ensureDir(directory) {
  await fs.mkdir(directory, {recursive: true});
}

async function writeJson(filePath, value) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function buildFrontMatter(doc) {
  const lines = ['---', `slug: ${JSON.stringify(doc.slug)}`];
  if (doc.className) lines.push(`className: ${JSON.stringify(doc.className)}`);
  if (doc.title) {
    lines.push(`title: ${JSON.stringify(doc.title)}`);
  }
  if (typeof doc.sidebarPosition === 'number') {
    lines.push(`sidebar_position: ${doc.sidebarPosition}`);
  }
  if (doc.sidebarLabel) {
    lines.push(`sidebar_label: ${JSON.stringify(doc.sidebarLabel)}`);
  }
  if (doc.dest === 'index.md') {
    lines.push('pagination_next: null', 'pagination_prev: null');
  }
  lines.push(
    `custom_edit_url: ${JSON.stringify(`${repoEditBaseUrl}/${doc.source}`)}`,
    '---',
    ''
  );
  return lines.join('\n');
}

export function resolveDocLink(fromSource, target, mappedAsRoute = false) {
  const ownSite = profile.siteUrl + profile.basePath;
  if (target?.startsWith(ownSite)) {
    return 'pathname://' + siteRoute(profile.basePath, '/' + target.slice(ownSite.length));
  }
  if (!target || /^(https?:|mailto:|#|\/)/.test(target)) {
    return target;
  }
  const [targetPath, targetHash] = target.split('#');
  const resolvedSource = normalizePath(
    path.posix.normalize(path.posix.join(path.posix.dirname(normalizePath(fromSource)), decodeURIComponent(targetPath)))
  );
  let targetDoc = mappingBySource.get(resolvedSource);
  if (!targetDoc && !path.posix.extname(resolvedSource)) {
    targetDoc = mappingBySource.get(normalizePath(path.posix.join(resolvedSource, 'README.md')));
  }
  if (!targetDoc) {
    const repoFsPath = path.resolve(repoRoot, resolvedSource);
    if (!isPathInside(repoRoot, repoFsPath) || !fsSync.existsSync(repoFsPath)) {
      return target;
    }
    const encodedSource = resolvedSource.split('/').map(encodeURIComponent).join('/');
    if (/\.(?:png|jpe?g|gif|webp|svg|pdf)$/i.test(resolvedSource)) {
      const route = `/repository/${encodedSource}`;
      return targetHash ? `${route}#${targetHash}` : route;
    }
    const repoUrlBase = fsSync.statSync(repoFsPath).isDirectory()
      ? repoTreeBaseUrl
      : repoBlobBaseUrl;
    return targetHash
      ? `${repoUrlBase}/${encodedSource}#${targetHash}`
      : `${repoUrlBase}/${encodedSource}`;
  }
  if (mappedAsRoute) {
    const route = siteRoute(profile.basePath, targetDoc.slug);
    return targetHash ? `${route}#${targetHash}` : route;
  }
  const currentDoc = mappingBySource.get(normalizePath(fromSource));
  const fromDest = currentDoc ? normalizePath(currentDoc.dest) : '';
  const toDest = normalizePath(targetDoc.dest);
  const relative = normalizePath(
    path.posix.relative(path.posix.dirname(fromDest), toDest)
  ) || path.posix.basename(toDest);
  return targetHash ? `${relative}#${targetHash}` : relative;
}

function rewriteLinks(content, fromSource) {
  const markdown = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, target) =>
    `[${label}](${resolveDocLink(fromSource, target.trim())})`
  );
  return markdown.replace(
    /(<a\b[^>]*?\bhref\s*=\s*)(["'])([^"']+)\2/gi,
    (match, prefix, quote, target) =>
      `${prefix}${quote}${resolveDocLink(fromSource, target.trim(), true)}${quote}`
  );
}

function rewriteOutsideFences(content, transform) {
  let output = '';
  let plain = '';
  let fence = null;
  for (const line of content.match(/[^\n]*\n|[^\n]+$/g) || []) {
    const match = line.match(/^\s*(`{3,}|~{3,})/);
    if (match) {
      const marker = match[1];
      if (fence === null) {
        output += transform(plain);
        plain = '';
        fence = marker;
      } else if (marker[0] === fence[0] && marker.length >= fence.length) {
        fence = null;
      }
      output += line;
      continue;
    }
    if (fence === null) {
      plain += line;
    } else {
      output += line;
    }
  }
  return output + transform(plain);
}

function rewriteOutsideInlineCode(content, transform) {
  let masked = '';
  let index = 0;
  const inlineCode = [];
  while (index < content.length) {
    const opening = content.indexOf('`', index);
    if (opening === -1) {
      masked += content.slice(index);
      break;
    }
    masked += content.slice(index, opening);
    let markerEnd = opening;
    while (content[markerEnd] === '`') {
      markerEnd += 1;
    }
    const marker = content.slice(opening, markerEnd);
    let closing = content.indexOf(marker, markerEnd);
    while (
      closing !== -1
      && (content[closing - 1] === '`' || content[closing + marker.length] === '`')
    ) {
      closing = content.indexOf(marker, closing + 1);
    }
    const spanEnd = closing === -1 ? content.length : closing + marker.length;
    const placeholder = `\u0000INLINE_CODE_${inlineCode.length}\u0000`;
    inlineCode.push(content.slice(opening, spanEnd));
    masked += placeholder;
    index = spanEnd;
  }
  return inlineCode.reduce(
    (result, code, codeIndex) =>
      result.replace(`\u0000INLINE_CODE_${codeIndex}\u0000`, () => code),
    transform(masked)
  );
}

export function rewriteOutsideCode(content, transform) {
  return rewriteOutsideFences(
    content,
    (chunk) => rewriteOutsideInlineCode(chunk, transform)
  );
}

function rewriteLinksOutsideCode(content, fromSource) {
  return rewriteOutsideFences(
    content,
    (chunk) => rewriteOutsideInlineCode(chunk, (plain) => rewriteLinks(plain, fromSource))
  );
}

export function normalizeForMdx(content, fromSource) {
  const rewrittenLinks = rewriteLinksOutsideCode(content, fromSource);
  return rewriteOutsideCode(rewrittenLinks, (chunk) =>
    chunk
      .replace(/<p align="center">([\s\S]*?)<\/p>/g, '<div align="center">$1</div>')
      .replace(/<([A-Za-z0-9.!#$%&'*+\/=?^_`{|}~-]+@[A-Za-z0-9.-]+)>/g, '[$1](mailto:$1)')
      .replace(/<(https?:\/\/[^<>\s]+)>/g, '[$1]($1)')
      .replace(/<br>/g, '<br />')
      .replace(/<hr>/g, '<hr />')
      .replace(/<img([^>]*?)(?<!\/)>/g, '<img$1 />')
      .replace(
        /\{(DISPLAY_NAME|ONE_SENTENCE_DESCRIPTION|REPOSITORY_NAME|REPOSITORY_OWNER)\}/g,
        '&#123;$1&#125;'
      )
  );
}

async function copyDoc(doc) {
  const sourcePath = resolveRepositoryFile(repoRoot, doc.source);
  const destPath = path.resolve(outRoot, doc.dest);
  const raw = await fs.readFile(sourcePath, 'utf8');
  const output = `${buildFrontMatter(doc)}${normalizeForMdx(raw, doc.source)}`;
  await ensureDir(path.dirname(destPath));
  await fs.writeFile(destPath, output);
}

async function writeCategoryFiles() {
  for (const category of sections) {
    await writeJson(path.resolve(outRoot, category.dir, '_category_.json'), {
      label: category.label,
      position: category.position,
      collapsible: true,
      collapsed: true,
      link: {
        type: 'generated-index',
        slug: category.slug,
        title: category.label,
        description: category.description,
      },
    });
  }
}

async function writeRobots() {
  await ensureDir(staticRoot);
  await fs.writeFile(
    path.resolve(staticRoot, 'robots.txt'),
    `User-agent: *\nAllow: /\nSitemap: ${canonicalSitemapUrl(
      profile.siteUrl,
      profile.basePath
    )}\n`
  );
}

async function copySourceAssets(directory = path.resolve(repoRoot, 'docs')) {
  for (const entry of await fs.readdir(directory, {withFileTypes: true})) {
    const source = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await copySourceAssets(source);
    } else if (/\.(?:png|jpe?g|gif|webp|svg|pdf)$/i.test(entry.name)) {
      const relative = path.relative(repoRoot, source);
      const destination = path.resolve(staticRoot, 'repository', relative);
      await ensureDir(path.dirname(destination));
      await fs.copyFile(source, destination);
    }
  }
}

async function main() {
  await fs.rm(path.resolve(siteRoot, '.generated'), {recursive: true, force: true});
  await ensureDir(outRoot);
  await writeCategoryFiles();
  for (const doc of documents) {
    await copyDoc(doc);
  }
  await fs.cp(path.resolve(siteRoot, 'static'), staticRoot, {recursive: true});
  await writeRobots();
  await copySourceAssets();
}

if (process.argv[1] === scriptPath) {
  await main();
}
