export function canonicalSitemapUrl(siteUrl, basePath) {
  const siteRoot = siteUrl.replace(/\/+$/, '');
  const docsRoot = basePath === '/' ? '/' : `${basePath.replace(/\/+$/, '')}/`;
  return `${siteRoot}${docsRoot}sitemap.xml`;
}

export function siteRoute(basePath, slug) {
  const docsRoot = basePath === '/' ? '' : basePath.replace(/\/+$/, '');
  return slug === '/' ? `${docsRoot}/` : `${docsRoot}${slug}`;
}
