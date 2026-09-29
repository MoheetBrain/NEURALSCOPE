import Link from "next/link";
import type { ButtonHTMLAttributes, ComponentProps } from "react";
import { cn } from "@/lib/cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "xs" | "sm" | "md" | "lg";

const BASE =
  "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-[var(--ns-radius)] border font-medium " +
  "transition-[background-color,border-color,color,box-shadow] duration-150 " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ns-accent " +
  "disabled:pointer-events-none disabled:opacity-45 aria-disabled:pointer-events-none aria-disabled:opacity-45";

const VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "border-ns-accent bg-ns-accent text-ns-bg hover:border-[#a9c4ff] hover:bg-[#a9c4ff] " +
    "shadow-[inset_0_1px_0_rgba(255,255,255,0.25)]",
  secondary:
    "border-ns-border-strong bg-ns-panel text-ns-text hover:border-ns-text-faint hover:bg-ns-panel-2",
  ghost: "border-transparent bg-transparent text-ns-text-muted hover:bg-ns-panel-2 hover:text-ns-text",
  danger: "border-ns-danger/40 bg-ns-danger/10 text-ns-danger hover:border-ns-danger/70 hover:bg-ns-danger/15",
};

const SIZES: Record<ButtonSize, string> = {
  xs: "h-6 px-2 [&_svg]:size-3",
  sm: "h-7 px-2.5 [&_svg]:size-3.5",
  md: "h-9 px-3.5 [&_svg]:size-4",
  lg: "h-11 px-5 [&_svg]:size-4",
};

// Font sizes are kept separate so `caps` can swap the whole type treatment without class conflicts.
const TEXT: Record<ButtonSize, string> = {
  xs: "font-sans text-[11px]",
  sm: "font-sans text-xs",
  md: "font-sans text-[13px]",
  lg: "font-sans text-sm",
};

const CAPS_TEXT: Record<ButtonSize, string> = {
  xs: "font-mono text-[10px] uppercase tracking-[0.08em]",
  sm: "font-mono text-[11px] uppercase tracking-[0.08em]",
  md: "font-mono text-[11.5px] uppercase tracking-[0.08em]",
  lg: "font-mono text-[12.5px] uppercase tracking-[0.08em]",
};

export interface ButtonStyleOptions {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Mono, uppercase, letter-spaced label (instrument style CTAs). */
  caps?: boolean;
  className?: string;
}

/** Class string for anything that should look like a button (links, labels...). */
export function buttonClasses({ variant = "secondary", size = "md", caps = false, className }: ButtonStyleOptions = {}): string {
  return cn(BASE, VARIANTS[variant], SIZES[size], caps ? CAPS_TEXT[size] : TEXT[size], className);
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  caps?: boolean;
}

export function Button({ variant = "secondary", size = "md", caps, className, type = "button", ...rest }: ButtonProps) {
  return <button type={type} className={buttonClasses({ variant, size, caps, className })} {...rest} />;
}

export type ButtonLinkProps = ComponentProps<typeof Link> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  caps?: boolean;
};

/** A next/link styled as a Button. */
export function ButtonLink({ variant = "secondary", size = "md", caps, className, ...rest }: ButtonLinkProps) {
  return <Link className={buttonClasses({ variant, size, caps, className })} {...rest} />;
}
