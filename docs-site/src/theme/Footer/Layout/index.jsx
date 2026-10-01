import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

export default function FooterLayout({links, copyright}) {
  const {siteConfig} = useDocusaurusContext();
  return (
    <footer className="footer">
      <div className="autonomio-wrapper">
        <Link to="/" className="autonomio-footer-wordmark">{siteConfig.themeConfig.navbar.title}</Link>
        {links}
        <div className="footer__copyright">{copyright}</div>
      </div>
    </footer>
  );
}
