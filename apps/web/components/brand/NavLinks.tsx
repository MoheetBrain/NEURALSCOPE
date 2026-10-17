"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

export const MARKETING_NAV = [
  { href: "/app", label: "Debugger", hideBelow: "lg" },
  { href: "/app/math", label: "Math Lab", hideBelow: "lg" },
  { href: "/app/selflab", label: "SelfLab", hideBelow: "sm" },
  { href: "/research", label: "Research", hideBelow: null },
  { href: "/docs", label: "Docs", hideBelow: null },
] as const;

// Static class strings so Tailwind can see them (keeps the header from overflowing on phones).
const HIDE_BELOW = { sm: "hidden sm:block", lg: "hidden lg:block" } as const;

export function NavLinks({ className }: { className?: string }) {
  const pathname = usePathname() ?? "/";
  return (
    <ul className={cn("flex items-center gap-1", className)}>
      {MARKETING_NAV.map((item) => {
        // "/app" is a prefix of its sub-workspaces, so it is only active on an exact match.
        const active = pathname === item.href || (item.href !== "/app" && pathname.startsWith(`${item.href}/`));
        return (
          <li key={item.href} className={item.hideBelow ? HIDE_BELOW[item.hideBelow] : undefined}>
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
