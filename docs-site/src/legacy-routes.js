import docsMap from '../docs-map.json';
import productDocs from '../product-docs.json';

// Docsify used hash routes. Keep bookmarked entries during the renderer change.
export function legacyDestination(hash) {
  if (!hash.startsWith('#/')) return null;
  const [legacyPath, query = ''] = hash.slice(1).split('?');
  let decoded;
  try { decoded = decodeURIComponent(legacyPath); } catch { return null; }
  const page = docsMap.documents.find((document) =>
    document.legacySlugs?.includes(decoded)
  );
  if (!page) return null;
  const fragment = new URLSearchParams(query).get('id');
  const route = productDocs.basePath.replace(/\/$/, '') + page.slug;
  return fragment ? `${route}#${encodeURIComponent(fragment.toLowerCase())}` : route;
}

export function onRouteDidUpdate() {
  const destination = legacyDestination(window.location.hash);
  if (destination) window.location.replace(destination);
}
