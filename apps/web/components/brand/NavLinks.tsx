"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

export const MARKETING_NAV = [
  { href: "/research", label: "Research" },
  { href: "/docs", label: "Docs" },
] as const;

export function NavLinks({ className }: { className?: string }) {
  const pathname = usePathname() ?? "/";
  return (
    <ul className={cn("flex items-center gap-1", className)}>
      {MARKETING_NAV.map((item) => {
        const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
        return (
          <li key={item.href}>
            <Link
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "inline-flex h-8 items-center rounded-[var(--ns-radius)] px-2.5 text-[13px] transition-colors",
                active ? "text-ns-text" : "text-ns-text-muted hover:text-ns-text",
              )}
            >
              {item.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
