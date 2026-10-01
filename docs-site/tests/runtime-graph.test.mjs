import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const manifest = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url)));
const lock = JSON.parse(fs.readFileSync(new URL('../package-lock.json', import.meta.url)));

test('Docusaurus providers and theme hooks resolve one framework version', () => {
  const expected = manifest.dependencies['@docusaurus/core'];
  for (const [packagePath, entry] of Object.entries(lock.packages)) {
    if (/(?:^|\/)node_modules\/@docusaurus\/[^/]+$/.test(packagePath)) {
      assert.equal(entry.version, expected, `${packagePath} introduces a second framework context`);
    }
  }
  for (const name of ['@docusaurus/plugin-content-docs', '@docusaurus/theme-common']) {
    assert.equal(manifest.dependencies[name], expected, `${name} must be an explicit aligned dependency`);
  }
});
