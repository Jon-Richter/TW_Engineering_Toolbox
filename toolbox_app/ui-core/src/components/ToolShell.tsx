import React, { ReactNode } from 'react';

interface ToolShellProps {
  title: string;
  eyebrow?: string;
  description?: string;
  tags?: Array<{ label: string }>;
  children: ReactNode;
}

export function ToolShell({
  title,
  eyebrow,
  description,
  tags = [],
  children,
}: ToolShellProps) {
  return (
    <div className="container">
      <header className="masthead">
        <div>
          {eyebrow && <div className="eyebrow">{eyebrow}</div>}
          <h1>{title}</h1>
          {description && <div className="sub">{description}</div>}
        </div>
        {tags.length > 0 && (
          <div className="masthead-tags">
            {tags.map((tag, idx) => (
              <div key={idx} className="tag">
                {tag.label}
              </div>
            ))}
          </div>
        )}
      </header>

      <main>{children}</main>
    </div>
  );
}
