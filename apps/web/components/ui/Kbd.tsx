import type { HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export type KbdProps = HTMLAttributes<HTMLElement>;

/** Keyboard key hint, e.g. <Kbd>⌘</Kbd><Kbd>K</Kbd>. */
export function Kbd({ className, ...rest }: KbdProps) {
  return (
    <kbd
      className={cn(
        "inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[3px] border border-ns-border-strong border-b-2",
        "bg-ns-panel-2 px-1 font-mono text-[10px] font-medium leading-none text-ns-text-muted",
        className,
      )}
      {...rest}
    />
  );
}
