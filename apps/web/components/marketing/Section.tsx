import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export function Container({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("mx-auto w-full max-w-[1280px] px-4 sm:px-6 lg:px-8", className)}>{children}</div>;
}

export interface SectionProps {
  id?: string;
  /** Margin index, e.g. "01". */
  index: string;
  /** Margin kicker, e.g. "Premise". */
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
  className?: string;
  children?: ReactNode;
}

/**
 * Marketing section with a research-paper style margin label
 * (§ index + kicker) and a large uppercase headline.
 */
export function Section({ id, index, kicker, title, lede, className, children }: SectionProps) {
  const headingId = id ? `${id}-title` : undefined;
  return (
    <section id={id} aria-labelledby={headingId} className={cn("border-t border-ns-border py-20 sm:py-24 lg:py-28", className)}>
      <Container>
        <div className="grid gap-y-6 lg:grid-cols-12 lg:gap-x-8">
          <div className="lg:col-span-3">
            <p className="flex items-center gap-3 pt-1">
              <span className="ns-label text-ns-text">§ {index}</span>
              <span aria-hidden="true" className="h-px w-8 bg-ns-border-strong" />
              <span className="ns-label">{kicker}</span>
            </p>
          </div>
          <div className="lg:col-span-9">
            <h2
              id={headingId}
              className="max-w-[30ch] text-balance text-[26px] font-normal uppercase leading-[1.12] tracking-[0.015em] text-ns-text sm:text-[34px] lg:text-[40px]"
            >
              {title}
            </h2>
            {lede ? <div className="mt-6 max-w-[62ch] space-y-4 text-[16px] leading-relaxed text-ns-text-muted sm:text-[17px]">{lede}</div> : null}
          </div>
        </div>
        {children ? <div className="mt-14 lg:mt-20">{children}</div> : null}
      </Container>
    </section>
  );
}
