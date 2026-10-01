const path = require('node:path');
const {themes: prismThemes} = require('prism-react-renderer');
const productDocs = require('./product-docs.json');

function repositoryCoordinates(sourceRepoUrl) {
  const url = new URL(sourceRepoUrl);
  const [organizationName, projectName] = url.pathname.split('/').filter(Boolean);
  return {organizationName, projectName};
}

const baseUrl = productDocs.basePath;
const url = productDocs.siteUrl;
const homeRouteRegex = `^${baseUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`;
const {organizationName, projectName} = repositoryCoordinates(productDocs.sourceRepoUrl);

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: productDocs.productName,
  tagline: productDocs.tagline,
  url,
  baseUrl,
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  trailingSlash: false,
  organizationName,
  projectName,
  themes: [],
  clientModules: [require.resolve('./src/legacy-routes.js')],
  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        docsRouteBasePath: '/',
        docsDir: '.generated/docs',
        indexDocs: true,
        indexBlog: false,
        hashed: true,
      },
    ],
  ],
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: path.resolve(__dirname, '.generated/docs'),
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: `${productDocs.sourceRepoUrl}/edit/${productDocs.sourceBranch}/`,
        },
        blog: false,
        pages: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  staticDirectories: [path.resolve(__dirname, '.generated/static')],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {respectPrefersColorScheme: true},
      metadata: [
        {name: 'description', content: productDocs.tagline},
        {property: 'og:type', content: 'website'},
        {property: 'og:site_name', content: productDocs.productName},
        {property: 'og:title', content: `${productDocs.productName} Docs`},
        {property: 'og:description', content: productDocs.tagline},
        {property: 'og:url', content: `${url}${baseUrl}`},
        {name: 'twitter:card', content: 'summary'},
        {name: 'twitter:title', content: `${productDocs.productName} Docs`},
        {name: 'twitter:description', content: productDocs.tagline},
      ],
      navbar: {
        title: productDocs.wordmark,
        items: [
          {to: '/', label: 'Home', position: 'left', activeBaseRegex: homeRouteRegex},
          {to: '/overview', label: 'Overview', position: 'left'},
          {to: '/guides', label: 'Guides', position: 'left'},
          {to: '/reference', label: 'Reference', position: 'left'},
          {to: '/developer', label: 'Developer', position: 'left'},
          {to: '/packages', label: 'Packages', position: 'left'},
          {href: productDocs.sourceRepoUrl, label: 'GitHub', position: 'right'},
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Overview', to: '/overview'},
              {label: 'Guides', to: '/guides'},
              {label: 'Reference', to: '/reference'},
              {label: 'Developer', to: '/developer'},
              {label: 'Packages', to: '/packages'},
            ],
          },
          {
            title: 'Product',
            items: [
              {label: 'Repository', href: productDocs.sourceRepoUrl},
              {label: productDocs.organizationName, href: `https://github.com/${organizationName}`},
            ],
          },
        ],
        copyright: `Copyright ${new Date().getFullYear()} ${productDocs.organizationName}.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      docs: {
        sidebar: {
          hideable: true,
        },
      },
    }),
};

module.exports = config;
