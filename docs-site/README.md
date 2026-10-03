# Documentation scaffold

`docs-site/` assembles the maintained Talos Markdown corpus into a static
Docusaurus site. It owns rendering and verification; public API behavior remains
owned by the Python package.

## Prerequisites

Use Node.js 20.18.1 or later and npm. Install the exact dependency graph from the
committed lockfile. Browser tests require Playwright Chromium, installed with
`npm --prefix docs-site exec -- playwright install chromium`.

## Build and verify

From the repository root:

```bash
npm --prefix docs-site ci
npm --prefix docs-site run security:audit
npm --prefix docs-site run check
```

The audit reports all production advisories and enforces the
[documentation dependency exception policy](../docs/Developer/Documentation-System.md#documentation-dependency-exceptions).

The check lints mapped Markdown, tests source assembly, verifies external links,
builds every route, proves sitemap/search/asset budgets, and runs browser and
accessibility checks. A failed stage blocks the check.

For a local preview, run `npm --prefix docs-site run serve` after building and
open `http://127.0.0.1:3100/talos/`. Publication, hosting headers, and a production
cutover are separate deployment work.

## Authored and generated files

`product-docs.json` owns identity, repository coordinates, and publication base
path. `docs-map.json` maps each maintained source once into one of Overview,
Guides, Reference, Developer, or Packages. Scripts and tests enforce that map;
`src/css/custom.css` implements the Autonomio visual style.

Never author or commit `.generated`, `.docusaurus`, `build`, `node_modules`,
test results, or browser reports. The previous Docsify navigation is replaced
by this single assembly path; original guide and reference sources remain.

## Maintenance boundary

Update the profile and map when adding a source. Update source prose where a
claim is owned; do not fork it in generated output. Framework examples are
proved by the Python documentation verifier, independently of this renderer.

Use the [documentation system](../docs/Developer/Documentation-System.md) for
composition, route, and proof requirements, and the
[documentation style](../docs/Developer/Documentation-Style.md) for typography,
color, and responsive behavior.
