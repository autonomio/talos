import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {canonicalSitemapUrl, siteRoute} from './site-urls.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const siteRoot = path.resolve(path.dirname(scriptPath), '..');
const buildRoot = path.resolve(siteRoot, 'build');
const profile = JSON.parse(
  await fs.readFile(path.resolve(siteRoot, 'product-docs.json'), 'utf8')
);
const docsMap = JSON.parse(
  await fs.readFile(path.resolve(siteRoot, 'docs-map.json'), 'utf8')
);
const maxJavaScriptBytes = 1_500_000;
const maxCssBytes = 400_000;

function routeFile(slug) {
  if (slug === '/') {
    return 'index.html';
  }
  return `${slug.replace(/^\//, '')}.html`;
}

const expectedRoutes = new Set([
  ...docsMap.sections.map((section) => routeFile(section.slug)),
  ...docsMap.documents.map((document) => routeFile(document.slug)),
  'search.html',
]);
for (const route of expectedRoutes) {
  await fs.access(path.resolve(buildRoot, route));
}

const sitemap = await fs.readFile(path.resolve(buildRoot, 'sitemap.xml'), 'utf8');
const actualSitemap = new Set([...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]));
const expectedSitemapRoutes = new Set([
  ...docsMap.sections.map((section) => section.slug),
  ...docsMap.documents.map((document) => document.slug),
  '/search',
].map((slug) => profile.siteUrl + siteRoute(profile.basePath, slug)));
if ([...expectedSitemapRoutes].some((route) => !actualSitemap.has(route)) || [...actualSitemap].some((route) => !expectedSitemapRoutes.has(route))) {
  throw new Error('sitemap URL set differs from canonical route authority');
}
const sitemapCount = actualSitemap.size;
if (sitemapCount !== expectedRoutes.size) {
  throw new Error(`sitemap has ${sitemapCount} URLs, expected ${expectedRoutes.size}`);
}
const robots = await fs.readFile(path.resolve(buildRoot, 'robots.txt'), 'utf8');
const expectedSitemap = canonicalSitemapUrl(profile.siteUrl, profile.basePath);
if (!robots.includes(expectedSitemap)) {
  throw new Error(`robots.txt must name ${expectedSitemap}`);
}
const searchIndex = await fs.readFile(path.resolve(buildRoot, 'search-index.json'), 'utf8');
const searchPayload = JSON.parse(searchIndex);
const indexedDocuments = searchPayload[0]?.documents;
if (!Array.isArray(indexedDocuments)) {
  throw new Error('search index does not contain a page-document collection');
}
const indexedRoutes = new Set(indexedDocuments.map((document) => document.u));
for (const document of docsMap.documents) {
  const route = siteRoute(profile.basePath, document.slug);
  if (!indexedRoutes.has(route)) {
    throw new Error(`search index does not contain mapped route ${route}`);
  }
}

const assetRoot = path.resolve(buildRoot, 'assets');
const assetPaths = [];
async function collect(directory) {
  for (const entry of await fs.readdir(directory, {withFileTypes: true})) {
    const entryPath = path.resolve(directory, entry.name);
    if (entry.isDirectory()) {
      await collect(entryPath);
    } else {
      assetPaths.push(entryPath);
    }
  }
}
await collect(assetRoot);
async function largestAsset(extension) {
  const matchingPaths = assetPaths.filter((file) => file.endsWith(extension));
  if (matchingPaths.length === 0) {
    throw new Error(`build contains no ${extension} assets`);
  }
  return Math.max(
    ...await Promise.all(
      matchingPaths.map(async (file) => (await fs.stat(file)).size)
    )
  );
}
const largestJavaScript = await largestAsset('.js');
const largestCss = await largestAsset('.css');
const assetSummary = (
  `js=${largestJavaScript}/${maxJavaScriptBytes}, `
  + `css=${largestCss}/${maxCssBytes}`
);
if (largestJavaScript > maxJavaScriptBytes || largestCss > maxCssBytes) {
  throw new Error(`asset budget exceeded: ${assetSummary}`);
}
process.stdout.write(
  `Build verified: ${expectedRoutes.size} routes, ${assetSummary}\n`
);
