import { clsx, type ClassValue } from "clsx";

/** Join conditional class names. (No tailwind-merge: keep class lists non-conflicting.) */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}
