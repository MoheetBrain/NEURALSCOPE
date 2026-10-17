import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

import LandingPage from "@/app/(marketing)/page";
import { FEATURE_GROUPS, FeatureStatusList } from "@/components/marketing/FeatureStatusList";
import { SelfLabExample } from "@/components/marketing/SelfLabSection";
import { NavLinks } from "@/components/brand/NavLinks";
import { SELFLAB_EXAMPLE, type SelfLabRecordedExample } from "@/lib/selflab-example";

describe("landing page", () => {
  it("leads with the new headline, tagline and subheadline and keeps the recorded hero trace", () => {
    render(<LandingPage />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/^NEURALSCOPE$/);
    expect(screen.getByText("Inspect the model. Audit its claims.")).toBeInTheDocument();
    expect(
      screen.getByText("An experimental workbench for tracing AI responses, testing self-reports, and comparing explanations against measured evidence."),
    ).toBeInTheDocument();
    // the hero trace and the debugger / Math Lab content are still there
    expect(document.querySelector("figure[data-stage]")).not.toBeNull();
    expect(screen.getByRole("heading", { name: /a microscope for mathematical reasoning/i })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /open selflab/i }).length).toBeGreaterThan(0);
  });

  it("shows the committed recorded example from a real (non-demo) run", () => {
    expect(SELFLAB_EXAMPLE).not.toBeNull();
    expect(SELFLAB_EXAMPLE?.source).toBe("recorded");
    expect(SELFLAB_EXAMPLE?.is_demo).toBe(false);
    expect(SELFLAB_EXAMPLE?.export_path).toMatch(/^docs\/selflab\/results\/exports\/.+\.json$/);
    render(<LandingPage />);
    expect(screen.queryByTestId("selflab-example-pending")).toBeNull();
    expect(screen.getAllByText(SELFLAB_EXAMPLE!.experiment_id, { exact: false }).length).toBeGreaterThan(0);
  });

  it("says the SelfLab pilot record is pending while no recorded example exists", () => {
    render(<SelfLabExample example={null} />);
    const pending = screen.getByTestId("selflab-example-pending");
    expect(pending).toHaveTextContent(/pilot record pending/i);
    expect(within(pending).getByRole("link", { name: /open selflab/i })).toHaveAttribute("href", "/app/selflab");
    // no numbers are invented for the pending state
    expect(pending.textContent).not.toMatch(/\d+(\.\d+)?\s*%/);
  });

  it("renders a recorded example with kind, status and scope kept apart and a fixture warning when demo", () => {
    const ex: SelfLabRecordedExample = {
      source: "recorded",
      experiment_id: "exp_test",
      template: "source_influence_battery",
      split: "debug_pilot",
      recorded_at: "2026-10-17T12:00:00Z",
      model_id: "demo/random-qwen2-tiny",
      is_demo: true,
      spec_sha256: "a".repeat(64),
      plan_sha256: "b".repeat(64),
      results: [{ label: "cue-following rate", value: "0.10 [0.02, 0.30]" }],
      evidence: [{ title: "Self-report vs causal source", kind: "model_report", status: "inconclusive", statement: "s", scope: "20 entities" }],
      limitations: ["fixture model"],
    };
    render(<SelfLabExample example={ex} />);
    const fig = screen.getByTestId("selflab-example");
    expect(fig).toHaveTextContent("Model report");
    expect(fig).toHaveTextContent("inconclusive");
    expect(fig).toHaveTextContent("20 entities");
    expect(fig).toHaveTextContent(/demo \/ fixture/i);
    expect(fig.textContent).not.toMatch(/validated/i);
  });

  it("lists available / prototype / fixture / API-only / unsupported / roadmap groups", () => {
    render(<FeatureStatusList />);
    for (const g of FEATURE_GROUPS) expect(screen.getByTestId(`feature-group-${g.key}`)).toBeInTheDocument();
    expect(screen.getByTestId("feature-group-unsupported")).toHaveTextContent(/LLM investigator/);
    expect(screen.getByTestId("feature-group-roadmap")).toHaveTextContent(/Sparse autoencoders/);
  });

  it("adds SelfLab to the marketing nav next to the debugger and Math Lab", () => {
    render(<NavLinks />);
    expect(screen.getByRole("link", { name: "SelfLab" })).toHaveAttribute("href", "/app/selflab");
    expect(screen.getByRole("link", { name: "Debugger" })).toHaveAttribute("href", "/app");
    expect(screen.getByRole("link", { name: "Math Lab" })).toHaveAttribute("href", "/app/math");
    expect(screen.getByRole("link", { name: "Research" })).toBeInTheDocument();
  });
});
