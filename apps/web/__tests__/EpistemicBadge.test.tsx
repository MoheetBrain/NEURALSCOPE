import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { EPISTEMIC, EPISTEMIC_ORDER, isEpistemicStatus } from "@/lib/epistemic";

const EXPECTED = {
  observed: { label: "Observed", colorClass: "text-ns-observed", hex: "#7cc7f5" },
  derived: { label: "Derived", colorClass: "text-ns-derived", hex: "#b39dfa" },
  inferred: { label: "Inferred", colorClass: "text-ns-inferred", hex: "#f2b45a" },
  causal: { label: "Causally tested", colorClass: "text-ns-causal", hex: "#5fd3a0" },
} as const;

describe("EpistemicBadge", () => {
  it.each(EPISTEMIC_ORDER)("renders the %s label, colour class and definition tooltip", (status) => {
    const { container } = render(<EpistemicBadge status={status} />);
    const badge = container.querySelector(`[data-status="${status}"]`);
    expect(badge).not.toBeNull();
    // label (rendered uppercase by CSS)
    expect(screen.getByText(EXPECTED[status].label, { exact: false })).toBeInTheDocument();
    // status-specific classes
    expect(badge).toHaveClass(`ns-epistemic-${status}`);
    expect(badge).toHaveClass(EXPECTED[status].colorClass);
    // tooltip carries the definition
    expect(badge?.getAttribute("title")).toContain(EPISTEMIC[status].definition);
    expect(badge?.getAttribute("title")?.startsWith(EXPECTED[status].label.toUpperCase())).toBe(true);
    // screen-reader prefix
    expect(badge).toHaveTextContent(`Epistemic status: ${EXPECTED[status].label}`);
  });

  it("does not leak other statuses' colour classes", () => {
    const { container } = render(<EpistemicBadge status="inferred" />);
    const badge = container.querySelector("[data-status]")!;
    for (const other of ["text-ns-observed", "text-ns-derived", "text-ns-causal"]) {
      expect(badge).not.toHaveClass(other);
    }
  });

  it("supports the xs size and a trailing note", () => {
    const { container } = render(<EpistemicBadge status="causal" size="xs" note="n=1" />);
    const badge = container.querySelector("[data-status]")!;
    expect(badge).toHaveClass("h-4");
    expect(badge).toHaveTextContent("n=1");
  });
});

describe("EPISTEMIC definitions", () => {
  it("defines the four statuses in ladder order with matching colour tokens", () => {
    expect(EPISTEMIC_ORDER).toEqual(["observed", "derived", "inferred", "causal"]);
    for (const s of EPISTEMIC_ORDER) {
      expect(EPISTEMIC[s].colorVar).toBe(`--ns-${s}`);
      expect(EPISTEMIC[s].color).toBe(`var(--ns-${s})`);
      expect(EPISTEMIC[s].hex).toBe(EXPECTED[s].hex);
      expect(EPISTEMIC[s].definition.length).toBeGreaterThan(20);
    }
  });

  it("narrows unknown values", () => {
    expect(isEpistemicStatus("causal")).toBe(true);
    expect(isEpistemicStatus("hypothesis")).toBe(false);
    expect(isEpistemicStatus(3)).toBe(false);
  });
});
