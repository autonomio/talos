import React from 'react';
import Link from '@docusaurus/Link';

export default function DocCardLayout({href, title, description}) {
  return (
    <Link to={href} className="autonomio-entry-row">
      <div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      <span aria-hidden="true">→</span>
    </Link>
  );
}
