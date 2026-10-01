import React from 'react';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import OriginalContent from '@theme-original/DocItem/Content';

export default function DocItemContent(props) {
  const {frontMatter} = useDoc();
  return (
    <div className={frontMatter.className}>
      <OriginalContent {...props} />
    </div>
  );
}
