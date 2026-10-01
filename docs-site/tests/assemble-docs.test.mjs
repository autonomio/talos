import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  normalizeForMdx,
  maintainedMarkdownSources,
  rewriteOutsideCode,
  validateDocuments,
  validateInventory,
  resolveDocLink,
  validateProfile,
  validateSections,
} from '../scripts/assemble-docs.mjs';
import {
  assertPublicUrl,
  extractExternalLinks,
  isPublicAddress,
} from '../scripts/check-external-links.mjs';
import {
  lintExitCode,
  markdownSources,
} from '../scripts/lint-markdown.mjs';
import {
  isPathInside,
  resolveRepositoryPath,
} from '../scripts/repository-paths.mjs';
import {canonicalSitemapUrl, siteRoute} from '../scripts/site-urls.mjs';

const mark = (value) => value.replaceAll('{TOKEN}', 'REWRITTEN');


test('rewrites prose but preserves fenced and inline code', () => {
  const source = [
    'before {TOKEN}',
    '```text',
    'inside {TOKEN}',
    '```',
    '~~~text',
    'inside tilde {TOKEN}',
    '~~~',
    'after `{TOKEN}` and {TOKEN}',
  ].join('\n');

  assert.equal(
    rewriteOutsideCode(source, mark),
    [
      'before REWRITTEN',
      '```text',
      'inside {TOKEN}',
      '```',
      '~~~text',
      'inside tilde {TOKEN}',
      '~~~',
      'after `{TOKEN}` and REWRITTEN',
    ].join('\n')
  );
});

test('preserves the tail of an unclosed fence', () => {
  const source = 'before {TOKEN}\n```text\ninside {TOKEN}';

  assert.equal(
    rewriteOutsideCode(source, mark),
    'before REWRITTEN\n```text\ninside {TOKEN}'
  );
});

test('preserves the tail of unmatched inline code', () => {
  const source = 'before {TOKEN} and `{TOKEN}';

  assert.equal(
    rewriteOutsideCode(source, mark),
    'before REWRITTEN and `{TOKEN}'
  );
});

test('preserves multi-backtick inline code while transforming surrounding prose', () => {
  const source = 'before {TOKEN}, ``inside $& $$ ` {TOKEN}``, after {TOKEN}';

  assert.equal(
    rewriteOutsideCode(source, mark),
    'before REWRITTEN, ``inside $& $$ ` {TOKEN}``, after REWRITTEN'
  );
});

test('extracts unique Markdown and HTML external links', () => {
  const source = [
    '[Repository](https://github.com/Autonomio/example)',
    '<a href="https://autonomio.github.io/example/">Docs</a>',
    '<img src="https://autonomio.github.io/example/logo.png" alt="Logo" />',
    '[Duplicate](https://github.com/Autonomio/example)',
    '`[Inline example](http://127.0.0.1/private)`',
    '~~~markdown',
    '[Fenced example](http://169.254.169.254/latest/meta-data)',
    '~~~',
  ].join('\n');

  assert.deepEqual(
    [...extractExternalLinks(source)].sort(),
    [
      'https://autonomio.github.io/example/',
      'https://autonomio.github.io/example/logo.png',
      'https://github.com/Autonomio/example',
    ]
  );
});

test('rewrites prose links without mutating code examples', () => {
  const source = [
    '[Docs](docs/README.md)',
    '[``Docs with `ticks` ``](docs/README.md)',
    '<a href="docs/README.md">Docs</a>',
    '```markdown',
    '[Docs](docs/README.md)',
    '```',
    '`[Docs](docs/README.md)`',
    '``[Docs](docs/README.md) with `ticks` ``',
  ].join('\n');

  const normalized = normalizeForMdx(source, 'README.md');
  assert.match(normalized, /^\[Docs\]\(overview\/docs-hub\.md\)$/m);
  assert.match(
    normalized,
    /^\[``Docs with `ticks` ``\]\(overview\/docs-hub\.md\)$/m
  );
  assert.match(
    normalized,
    /^<a href="\/(?:[^"]*\/)?overview\/docs-hub">Docs<\/a>$/m
  );
  assert.match(normalized, /^`\[Docs\]\(docs\/README\.md\)`$/m);
  assert.match(normalized, /^``\[Docs\]\(docs\/README\.md\) with `ticks` ``$/m);
  assert.match(
    normalized,
    /```markdown\n\[Docs\]\(docs\/README\.md\)\n```/
  );
});

test('keeps repository link resolution inside the repository root', () => {
  assert.equal(isPathInside('/repo', '/repo'), true);
  assert.equal(isPathInside('/repo', '/repo/docs/README.md'), true);
  assert.equal(isPathInside('/repo', '/repo-neighbor/README.md'), false);
  assert.equal(isPathInside('/repo', '/outside/README.md'), false);
  assert.throws(
    () => resolveRepositoryPath('/repo', '../outside/README.md'),
    /documentation source is outside the repository/
  );
});

test('builds canonical sitemap URLs for root and nested docs paths', () => {
  assert.equal(
    canonicalSitemapUrl('https://docs.example.com', '/'),
    'https://docs.example.com/sitemap.xml'
  );
  assert.equal(
    canonicalSitemapUrl('https://docs.example.com/', '/product/'),
    'https://docs.example.com/product/sitemap.xml'
  );
  assert.equal(siteRoute('/', '/guide'), '/guide');
  assert.equal(siteRoute('/', '/'), '/');
  assert.equal(siteRoute('/product/', '/guide'), '/product/guide');
  assert.equal(siteRoute('/product/', '/'), '/product/');
});

test('derives Markdown lint sources from the route map', () => {
  const map = {
    documents: [
      {source: 'README.md'},
      {source: 'docs/README.md'},
      {source: 'README.md'},
    ],
  };

  assert.deepEqual(
    markdownSources(map),
    ['README.md', 'docs/README.md']
  );
});

test('fails markdown lint when the subprocess exits by signal', () => {
  assert.equal(lintExitCode(null), 1);
  assert.equal(lintExitCode(2), 2);
});

test('validates every category field before generation', () => {
  const sections = Array.from({length: 5}, (_, index) => ({
    dir: `section-${index}`,
    label: `Section ${index}`,
    position: index + 1,
    slug: `/section-${index}`,
    description: `Section ${index} description.`,
  }));

  assert.doesNotThrow(() => validateSections(sections));
  assert.throws(
    () => validateSections([{...sections[0], label: ''}, ...sections.slice(1)]),
    /section.label must be a non-empty string/
  );
  assert.throws(
    () => validateSections([{...sections[0], position: 0}, ...sections.slice(1)]),
    /section.position must be a positive integer/
  );
  assert.throws(
    () => validateSections([{...sections[0], slug: '/section/'}, ...sections.slice(1)]),
    /section.slug must be \/ or a canonical leading-slash route/
  );
});

test('validates canonical document routes before generation', () => {
  const documents = [
    {source: 'README.md', dest: 'index.md', slug: '/'},
    {source: 'docs/README.md', dest: 'overview.md', slug: '/overview'},
  ];

  assert.doesNotThrow(() => validateDocuments(documents));
  assert.throws(
    () => validateDocuments([{...documents[0], slug: 'overview'}]),
    /document.slug must be \/ or a canonical leading-slash route/
  );
  assert.throws(
    () => validateDocuments([{...documents[0], slug: '/overview/'}]),
    /document.slug must be \/ or a canonical leading-slash route/
  );
});

test('validates documentation deployment coordinates before generation', () => {
  const profile = {
    productId: 'product',
    productName: 'Product',
    tagline: 'Product documentation.',
    siteUrl: 'https://docs.example.com',
    basePath: '/product/',
    sourceRepoUrl: 'https://github.com/Autonomio/product',
  };

  assert.doesNotThrow(() => validateProfile(profile));
  assert.doesNotThrow(() => validateProfile({...profile, basePath: '/'}));
  assert.throws(
    () => validateProfile({...profile, siteUrl: 'https://docs.example.com/'}),
    /siteUrl must be an HTTP\(S\) origin without a trailing slash/
  );
  assert.throws(
    () => validateProfile({...profile, basePath: 'product/'}),
    /basePath must be \/ or have leading and trailing slashes/
  );
});

test('rejects external-link destinations that can reach private networks', async () => {
  assert.equal(isPublicAddress('8.8.8.8'), true);
  assert.equal(isPublicAddress('127.0.0.1'), false);
  assert.equal(isPublicAddress('169.254.169.254'), false);
  assert.equal(isPublicAddress('10.20.30.40'), false);
  assert.equal(isPublicAddress('192.0.2.1'), false);
  assert.equal(isPublicAddress('::1'), false);
  assert.equal(isPublicAddress('fe80::1'), false);

  await assert.rejects(
    assertPublicUrl('http://localhost/private'),
    /non-public destination/
  );
  await assert.rejects(
    assertPublicUrl('http://169.254.169.254/latest/meta-data'),
    /non-public destination/
  );
  await assert.rejects(
    assertPublicUrl('https://user:secret@example.com/'),
    /must not contain credentials/
  );
});


test('mapped inventory cannot silently omit or invent an authored page', () => {
  const documents = [{source: 'docs/a.md'}];
  assert.doesNotThrow(() => validateInventory(documents, ['docs/a.md']));
  assert.throws(() => validateInventory(documents, ['docs/a.md', 'docs/b.md']), /missing=docs\/b.md/);
  assert.throws(() => validateInventory(documents, []), /unexpected=docs\/a.md/);
});

test('local images resolve to published assets rather than repository HTML', () => {
  assert.equal(resolveDocLink('docs/Workflow.md', '_media/talos_deep_learning_workflow.png'), '/repository/docs/_media/talos_deep_learning_workflow.png');
});

test('legacy bookmark routes have one canonical destination', () => {
  const first = {source: 'a.md', dest: 'a.md', slug: '/a', legacySlugs: ['/Old']};
  assert.throws(() => validateDocuments([first, {...first, source: 'b.md', dest: 'b.md', slug: '/b'}]), /legacy slug values must be unique/);
});


test('external links retain balanced URL parentheses', () => {
  const url = 'https://example.com/dataset/(diagnostic)';
  assert.deepEqual([...extractExternalLinks(`[Dataset](${url})`)], [url]);
});


test('Markdown autolinks remain links in MDX without changing fenced examples', () => {
  const source = '<contact@example.com> and <https://example.com>\n\n```text\n<contact@example.com>\n```';
  const output = normalizeForMdx(source, 'README.md');
  assert.match(output, /\[contact@example.com\]\(mailto:contact@example.com\)/);
  assert.match(output, /\[https:\/\/example.com\]\(https:\/\/example.com\)/);
  assert.match(output, /```text\n<contact@example.com>\n```/);
});


test('new repository contracts enter the maintained inventory automatically', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'talos-doc-inventory-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  fs.mkdirSync(path.join(root, 'docs', 'Developer'), {recursive: true});
  fs.writeFileSync(path.join(root, 'README.md'), '# Talos\n');
  fs.writeFileSync(path.join(root, 'GOVERNANCE.md'), '# Governance\n');
  fs.writeFileSync(path.join(root, 'docs', 'Developer', 'Configuration.md'), '# Configuration\n');
  const sources = maintainedMarkdownSources(root);
  assert.ok(sources.includes('GOVERNANCE.md'));
  assert.ok(sources.includes('docs/Developer/Configuration.md'));
  assert.throws(() => validateInventory([{source: 'README.md'}], sources), /GOVERNANCE.md/);
});
