const {defineConfig} = require('@playwright/test');
const productDocs = require('./product-docs.json');

module.exports = defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  outputDir: './test-results',
  reporter: [['list']],
  timeout: 60000,
  workers: 2,
  use: {
    baseURL: `http://127.0.0.1:3100${productDocs.basePath}`,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run serve',
    url: `http://127.0.0.1:3100${productDocs.basePath}`,
    reuseExistingServer: false,
    timeout: 60000,
  },
});
