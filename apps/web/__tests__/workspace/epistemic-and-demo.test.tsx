import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DEMO_BANNER_TEXT, DemoBanner, demoReasons } from "@/components/workspace/DemoBanner";
import { StatusBadge } from "@/components/workspace/ui";
import { toEpistemic } from "@/lib/api/status";

describe("API status → epistemic badge", () => {
  it('maps "causal" to CAUSALLY TESTED', () => {
    expect(toEpistemic("causal")).toBe("causal");
    const { container } = render(<StatusBadge status="causal" note="single prompt (n=1)" />);
    const badge = container.querySelector('[data-status="causal"]');
    expect(badge).not.toBeNull();
    expect(badge).toHaveTextContent("Causally tested");
    expect(badge).toHaveTextContent("single prompt (n=1)");
    expect(badge).toHaveClass("text-ns-causal");
  });

  it("maps observed / derived / inferred one-to-one", () => {
    expect(toEpistemic("observed")).toBe("observed");
    expect(toEpistemic("derived")).toBe("derived");
    expect(toEpistemic("inferred")).toBe("inferred");
    render(<StatusBadge status="observed" />);
    expect(screen.getByText("Observed", { exact: false })).toBeInTheDocument();
  });

  it("never upgrades an unknown status to causal", () => {
    expect(toEpistemic("tested?")).toBe("derived");
    expect(toEpistemic(undefined)).toBe("derived");
    expect(toEpistemic(null)).toBe("derived");
  });
});

describe("DEMO banner", () => {
  it("renders the unmistakable demo warning when visible", () => {
    render(<DemoBanner visible reason="backend reports is_demo" />);
    const banner = screen.getByTestId("demo-banner");
    expect(banner).toHaveAttribute("role", "alert");
    expect(banner).toHaveTextContent("DEMO — RANDOM-WEIGHT FIXTURE MODEL · outputs carry no scientific meaning");
    expect(banner).toHaveTextContent("backend reports is_demo");
    expect(banner).toHaveClass("text-ns-warn");
  });

  it("renders nothing for real models", () => {
    const { container } = render(<DemoBanner visible={false} />);
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText(DEMO_BANNER_TEXT)).toBeNull();
  });

  it("derives visibility from health, model and run flags", () => {
    expect(demoReasons({ healthDemo: false, modelDemo: false, runDemo: false })).toEqual([]);
    expect(demoReasons({ healthDemo: true })).toEqual(["backend reports is_demo"]);
    expect(demoReasons({ modelDemo: true })).toEqual(["model reports is_demo"]);
    expect(demoReasons({ healthDemo: false, runDemo: true })).toEqual(["this run was recorded on the demo model"]);
  });
});
