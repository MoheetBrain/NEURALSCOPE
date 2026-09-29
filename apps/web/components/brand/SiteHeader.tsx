import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { ButtonLink } from "@/components/ui/Button";
import { StatusChip } from "@/components/ui/StatusChip";
import { Logo } from "./Logo";
import { NavLinks } from "./NavLinks";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-ns-border bg-ns-bg/85 backdrop-blur-md supports-[backdrop-filter]:bg-ns-bg/70">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-[var(--ns-radius)] focus:bg-ns-panel-2 focus:px-3 focus:py-1.5 focus:text-sm"
      >
        Skip to content
      </a>
      <div className="mx-auto flex h-[var(--ns-header-h)] max-w-[1280px] items-center gap-3 px-4 sm:px-6 lg:px-8">
        <Link href="/" aria-label="NeuralScope home" className="flex items-center gap-3 rounded-[var(--ns-radius)]">
          <Logo size={22} />
        </Link>
        <span className="hidden md:block">
          <StatusChip tone="neutral">V1 · Research preview</StatusChip>
        </span>
        <nav aria-label="Primary" className="ml-auto">
          <NavLinks />
        </nav>
        <div className="hidden sm:block">
          <ButtonLink href="/app" variant="primary" size="sm" caps>
            Open Neural Debugger
            <ArrowRight aria-hidden="true" />
          </ButtonLink>
        </div>
      </div>
    </header>
  );
}
