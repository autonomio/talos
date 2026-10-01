import React, {useId} from 'react';
import {useCollapsible, Collapsible} from '@docusaurus/theme-common';
import TOCItems from '@theme/TOCItems';
import CollapseButton from '@theme/TOCCollapsible/CollapseButton';

export default function TOCCollapsible({toc, className, minHeadingLevel, maxHeadingLevel}) {
  const {collapsed, toggleCollapsed} = useCollapsible({initialState: true});
  const panelId = useId();
  return (
    <div className={`autonomio-toc ${className || ''}`}>
      <CollapseButton collapsed={collapsed} onClick={toggleCollapsed}
        aria-expanded={!collapsed} aria-controls={panelId} />
      <Collapsible lazy id={panelId} collapsed={collapsed}>
        <TOCItems toc={toc} minHeadingLevel={minHeadingLevel} maxHeadingLevel={maxHeadingLevel} />
      </Collapsible>
    </div>
  );
}
