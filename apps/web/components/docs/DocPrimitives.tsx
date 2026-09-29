import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

/** Anchored section heading used by long-form pages (docs, research). */
export function DocHeading({
  id,
  level = 2,
  kicker,
  children,
  className,
}: {
  id: string;
  level?: 2 | 3;
  kicker?: string;
  children: ReactNode;
  className?: string;
}) {
  const Tag = level === 2 ? "h2" : "h3";
  return (
    <div className={cn("group", level === 2 ? "mb-6" : "mb-3 mt-12", className)}>
      {kicker ? <p className="ns-label mb-3">{kicker}</p> : null}
      <Tag
        id={id}
        className={cn(
          "scroll-mt-24 text-ns-text",
          level === 2 ? "text-[26px] font-medium leading-tight sm:text-[30px]" : "text-[18px] font-medium leading-snug",
        )}
      >
        {children}
        <a
          href={`#${id}`}
          aria-label="Link to this section"
          className="ml-2 font-mono text-[0.7em] text-ns-text-faint opacity-0 transition-opacity hover:text-ns-accent focus-visible:opacity-100 group-hover:opacity-100"
        >
          #
        </a>
      </Tag>
    </div>
  );
}

export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "max-w-[68ch] space-y-4 text-[15.5px] leading-[1.7] text-ns-text-muted",
        "[&_strong]:font-medium [&_strong]:text-ns-text",
        "[&_a]:text-ns-accent [&_a]:underline-offset-4 hover:[&_a]:underline",
        "[&_code]:rounded-[3px] [&_code]:border [&_code]:border-ns-border [&_code]:bg-ns-panel [&_code]:px-1 [&_code]:py-px [&_code]:font-mono [&_code]:text-[0.86em] [&_code]:text-ns-text [&_code]:[overflow-wrap:anywhere]",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CodeBlock({
  children,
  label,
  className,
}: {
  children: string;
  label?: string;
  className?: string;
}) {
  return (
    <figure className={cn("min-w-0 overflow-hidden rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel", className)}>
      {label ? (
        <figcaption className="flex h-8 items-center border-b border-ns-border px-3">
          <span className="ns-label">{label}</span>
        </figcaption>
      ) : null}
      <pre className="overflow-x-auto p-4 font-mono text-[12.5px] leading-relaxed text-ns-text">
        <code>{children}</code>
      </pre>
    </figure>
  );
}

/** Centered math/formula line in mono. */
export function Formula({ children, note }: { children: ReactNode; note?: ReactNode }) {
  return (
    <div className="my-5 max-w-[68ch] overflow-x-auto rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel/60 px-4 py-3">
      <p className="whitespace-nowrap font-mono text-[13.5px] text-ns-text">{children}</p>
      {note ? <p className="mt-1.5 font-mono text-[11px] text-ns-text-faint">{note}</p> : null}
    </div>
  );
}

export function Callout({
  tone = "neutral",
  title,
  children,
}: {
  tone?: "neutral" | "warn" | "accent";
  title: string;
  children: ReactNode;
}) {
  return (
    <aside
      className={cn(
        "my-6 max-w-[68ch] rounded-[var(--ns-radius)] border-l-2 bg-ns-panel px-4 py-3",
        tone === "neutral" && "border-ns-border-strong",
        tone === "warn" && "border-ns-warn",
        tone === "accent" && "border-ns-accent",
      )}
    >
      <p
        className={cn(
          "ns-label",
          tone === "warn" && "text-ns-warn",
          tone === "accent" && "text-ns-accent",
        )}
      >
        {title}
      </p>
      <div className="mt-2 text-[14px] leading-relaxed text-ns-text-muted [&_code]:font-mono [&_code]:text-[0.9em] [&_code]:text-ns-text [&_code]:[overflow-wrap:anywhere]">
        {children}
      </div>
    </aside>
  );
}

export interface TocItem {
  id: string;
  label: string;
  children?: { id: string; label: string }[];
}

export function Toc({ items, title = "On this page" }: { items: TocItem[]; title?: string }) {
  return (
    <nav aria-label={title} className="text-[13px]">
      <p className="ns-label mb-4">{title}</p>
      <ol className="space-y-1 border-l border-ns-border">
        {items.map((it, i) => (
          <li key={it.id}>
            <a
              href={`#${it.id}`}
              className="-ml-px flex gap-2 border-l border-transparent py-1 pl-3 text-ns-text-muted transition-colors hover:border-ns-text-muted hover:text-ns-text"
            >
              <span className="font-mono text-[11px] text-ns-text-faint">{String(i + 1).padStart(2, "0")}</span>
              {it.label}
            </a>
            {it.children ? (
              <ol className="mb-1 space-y-0.5">
                {it.children.map((c) => (
                  <li key={c.id}>
                    <a
                      href={`#${c.id}`}
                      className="-ml-px block border-l border-transparent py-0.5 pl-[42px] text-[12.5px] text-ns-text-faint transition-colors hover:border-ns-text-muted hover:text-ns-text"
                    >
                      {c.label}
                    </a>
                  </li>
                ))}
              </ol>
            ) : null}
          </li>
        ))}
      </ol>
    </nav>
  );
}

/** Page header for long-form pages. */
export function PageHeader({
  kicker,
  title,
  children,
}: {
  kicker: string;
  title: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="relative overflow-hidden border-b border-ns-border">
      <div
        aria-hidden="true"
        className="ns-grid-bg pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_70%_90%_at_20%_0%,black,transparent_70%)]"
      />
      <div className="relative mx-auto max-w-[1280px] px-4 pb-14 pt-16 sm:px-6 sm:pb-20 sm:pt-24 lg:px-8">
        <p className="flex items-center gap-3">
          <span aria-hidden="true" className="h-px w-8 bg-ns-accent" />
          <span className="ns-label">{kicker}</span>
        </p>
        <h1 className="mt-6 max-w-[24ch] text-balance text-[34px] font-light leading-[1.1] text-ns-text sm:text-[48px] lg:text-[56px]">
          {title}
        </h1>
        {children ? <div className="mt-6 max-w-[64ch] space-y-4 text-[17px] leading-relaxed text-ns-text-muted">{children}</div> : null}
      </div>
    </header>
  );
}
