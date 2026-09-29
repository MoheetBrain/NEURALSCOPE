import Link from "next/link";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { EPISTEMIC_ORDER } from "@/lib/epistemic";
import { Logo } from "./Logo";

const LINKS = [
  {
    heading: "Product",
    items: [
      { href: "/app", label: "Neural Debugger" },
      { href: "/#math-lab", label: "Math Lab" },
      { href: "/#roadmap", label: "Roadmap" },
    ],
  },
  {
    heading: "Research",
    items: [
      { href: "/research", label: "Philosophy" },
      { href: "/research#first-question", label: "First question" },
      { href: "/research#experiments", label: "Planned experiments" },
    ],
  },
  {
    heading: "Docs",
    items: [
      { href: "/docs#quickstart", label: "Quickstart" },
      { href: "/docs#concepts", label: "Concepts" },
      { href: "/docs#limitations", label: "Limitations" },
    ],
  },
] as const;

export function SiteFooter() {
  return (
    <footer className="border-t border-ns-border bg-ns-bg">
      <div className="mx-auto grid max-w-[1280px] gap-10 px-4 py-14 sm:px-6 md:grid-cols-[1.4fr_repeat(3,1fr)] lg:px-8">
        <div className="max-w-sm">
          <Logo size={22} />
          <p className="mt-4 text-sm leading-relaxed text-ns-text-muted">
            A causal debugger for neural networks. Recover the algorithms that learning discovered.
          </p>
          <div className="mt-5 flex flex-wrap gap-1.5" aria-label="Epistemic status legend">
            {EPISTEMIC_ORDER.map((s) => (
              <EpistemicBadge key={s} status={s} size="xs" />
            ))}
          </div>
        </div>
        {LINKS.map((col) => (
          <nav key={col.heading} aria-label={col.heading}>
            <h2 className="ns-label">{col.heading}</h2>
            <ul className="mt-4 space-y-2.5">
              {col.items.map((it) => (
                <li key={it.href}>
                  <Link href={it.href} className="text-sm text-ns-text-muted transition-colors hover:text-ns-text">
                    {it.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        ))}
      </div>
      <div className="border-t border-ns-border">
        <div className="mx-auto flex max-w-[1280px] flex-col gap-2 px-4 py-5 font-mono text-[11px] text-ns-text-faint sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <span>© {new Date().getFullYear()} NeuralScope · V1 research preview</span>
          <span>Measurements are labelled. Hypotheses are marked. Claims are tested.</span>
        </div>
      </div>
    </footer>
  );
}
