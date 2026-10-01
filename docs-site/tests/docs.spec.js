const fs = require('node:fs');
const path = require('node:path');
const {test, expect} = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;
const docsMap = require('../docs-map.json');
const productDocs = require('../product-docs.json');

const contract = docsMap.documents.find((document) =>
  document.source === 'docs/Developer/Documentation-System.md'
);
const contractSource = fs.readFileSync(path.resolve(__dirname, '..', '..', contract.source), 'utf8');
const contractTitle = contractSource.match(/^# (.+)$/m)[1];
const navigation = ['Home', 'Overview', 'Guides', 'Reference', 'Developer', 'Packages', 'GitHub'];
const specimens = [
  {role: 'home', source: 'README.md'},
  {role: 'category', slug: '/guides', title: 'Guides'},
  {role: 'guide', source: 'docs/Guides/Quickstart.md'},
  {role: 'reference', source: 'docs/Scan.md'},
  {role: 'developer', source: contract.source},
  {role: 'package', source: 'talos/README.md'},
].map((specimen) => {
  const document = docsMap.documents.find((entry) => entry.source === specimen.source);
  return {...document, ...specimen};
});

for (const viewport of [{width: 1440, height: 900}, {width: 390, height: 844}]) {
  for (const theme of ['light', 'dark']) {
    for (const specimen of specimens) {
      test(`${specimen.role}: ${viewport.width}px ${theme}`, async ({page}, testInfo) => {
        await page.setViewportSize(viewport);
        const route = specimen.slug.replace(/^\//, '');
        const consoleErrors = [];
        page.on('pageerror', (error) => consoleErrors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
        const response = await page.goto(`${route}?docusaurus-theme=${theme}`);
        expect(response.status()).toBe(200);
        await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
        await expect(page.locator('main h1')).toBeVisible();
        await expect(page.locator('.navbar__link')).toHaveText(navigation);
        const activeSection = specimen.role === 'home' ? 'Home'
          : docsMap.sections.find((section) => specimen.slug === section.slug || specimen.slug.startsWith(`${section.slug}/`)).label;
        await expect(page.locator('.navbar__link--active')).toHaveText([activeSection]);
        for (const link of await page.locator('.navbar__link').all()) await expect(link).toBeVisible();
        await expect(page.locator('.navbar__toggle')).toHaveCount(0);
        await expect(page.locator('.navbar__title')).toHaveText(productDocs.wordmark);
        if (specimen.source) {
          await expect(page.getByText('Edit this page', {exact: true})).toHaveAttribute(
            'href', `${productDocs.sourceRepoUrl}/edit/${productDocs.sourceBranch}/${specimen.source}`
          );
        }
        const canonical = productDocs.siteUrl + productDocs.basePath.replace(/\/$/, '') + specimen.slug;
        await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', canonical);
        const geometry = await page.evaluate(() => {
          const article = document.querySelector('.theme-doc-markdown') || document.querySelector('main');
          const brand = document.querySelector('.navbar__title');
          return {
            width: article.getBoundingClientRect().width,
            font: getComputedStyle(article).fontFamily,
            bodySize: getComputedStyle(article).fontSize,
            headingSize: getComputedStyle(document.querySelector('main h1')).fontSize,
            fontReady: document.fonts.check('19px Finlandica'),
            navbar: document.querySelector('.navbar').getBoundingClientRect().height,
            brandSize: getComputedStyle(brand).fontSize,
            brandWeight: getComputedStyle(brand).fontWeight,
            overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          };
        });
        if (specimen.source) expect(geometry.width).toBeLessThanOrEqual(680);
        expect(geometry.font).toContain('Finlandica');
        expect(geometry.bodySize).toBe(viewport.width === 390 ? '18px' : '19px');
        expect(geometry.brandSize).toBe(viewport.width === 390 ? '24px' : '25px');
        expect(geometry.brandWeight).toBe('400');
        expect(geometry.fontReady).toBe(true);
        expect(geometry.headingSize).toBe(specimen.role === 'home'
          ? (viewport.width === 390 ? '74px' : '91px')
          : (viewport.width === 390 ? '65px' : '94px'));
        if (viewport.width === 1440) expect(geometry.navbar).toBe(142);
        expect(geometry.overflow).toBeLessThanOrEqual(1);
        const accessibility = await new AxeBuilder({page}).withTags(['wcag2a', 'wcag2aa']).analyze();
        expect(accessibility.violations).toEqual([]);
        expect(consoleErrors).toEqual([]);
        await page.screenshot({path: testInfo.outputPath('surface.png')});
        await testInfo.attach('surface', {path: testInfo.outputPath('surface.png'), contentType: 'image/png'});
      });
    }
  }
}

test('local search, keyboard navigation, theme, and code copy remain functional', async ({page, context}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('?docusaurus-theme=light');
  const search = page.locator('input[aria-label="Search"]');
  await search.fill(contractTitle);
  await expect(page.locator('[role="listbox"]')).toContainText(contractTitle);
  await search.press('ArrowDown');
  await search.press('Enter');
  await expect(page).toHaveURL(new RegExp(contract.slug));
  await expect(page.locator('h1')).toHaveText(contractTitle);
  await page.keyboard.press('Tab');
  expect(await page.evaluate(() => document.activeElement?.tagName)).not.toBe('BODY');
  const block = page.locator('.theme-code-block').first();
  await block.hover();
  const copy = block.getByRole('button', {name: /copy code/i});
  await copy.click();
  const clipboard = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboard).toContain('npm --prefix docs-site');
  await page.setViewportSize({width: 390, height: 844});
  await page.goto('packages/talos?docusaurus-theme=dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  const mobileSearch = page.locator('input[aria-label="Search"]');
  await expect(mobileSearch).toHaveAttribute('placeholder', 'Search');
  expect((await mobileSearch.boundingBox()).width).toBe(156);
  await mobileSearch.fill(contractTitle);
  await expect(page.locator('[role="listbox"]')).toContainText(contractTitle);
  await mobileSearch.press('ArrowDown');
  await mobileSearch.press('Enter');
  await expect(page).toHaveURL(new RegExp(contract.slug));
  await expect(page.locator('h1')).toHaveText(contractTitle);
  await expect(page.locator('.theme-doc-sidebar-container')).toBeHidden();
  const contents = page.locator('.theme-doc-toc-mobile button');
  await expect(contents).toBeVisible();
  await contents.click();
  await expect(contents).toHaveAttribute('aria-expanded', 'true');
  await expect(page.locator('.theme-doc-toc-mobile a').first()).toBeVisible();
});

test('existing Docsify bookmarks retain their reader destination', async ({page}) => {
  await page.goto('#/Scan?id=the-details');
  await expect(page).toHaveURL(/\/talos\/reference\/scan#the-details$/);
  await expect(page.locator('#the-details')).toHaveCount(1);
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(0);
  await page.goto('/talos/#/README');
  await expect(page).toHaveURL(/\/talos\/guides\/quickstart$/);
});


test('canonical source assets remain available in the rendered site', async ({request}) => {
  const diagram = await request.get('repository/docs/_media/talos_deep_learning_workflow.png');
  expect(diagram.ok()).toBe(true);
  expect(diagram.headers()['content-type']).toContain('image/png');
  const guide = await request.get('repository/docs/_media/autonomio-style-guide.pdf');
  expect(guide.ok()).toBe(true);
  expect(guide.headers()['content-type']).toContain('application/pdf');
  expect((await guide.body()).subarray(0, 4).toString()).toBe('%PDF');
});


test('the theme control offers light, dark, and system preference', async ({page}) => {
  await page.emulateMedia({colorScheme: 'light'});
  await page.goto('developer/documentation-system');
  await page.evaluate(() => localStorage.removeItem('theme'));
  await page.reload();
  const toggle = page.locator('.autonomio-tools button[title="system mode"]');
  await expect(toggle).toBeVisible();
  await toggle.focus();
  await toggle.press('Enter');
  await expect(page.locator('.autonomio-tools button[title="light mode"]')).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.locator('.autonomio-tools button[title="light mode"]').press('Enter');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.locator('.autonomio-tools button[title="dark mode"]').press('Enter');
  await expect(page.locator('.autonomio-tools button[title="system mode"]')).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
});
