import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { Tabs } from "@/components/ui/Tabs";

describe("Panel", () => {
  it("renders a labelled header with badge and right slot", () => {
    render(
      <Panel label="Logit Lens" status="derived" right={<span>L14</span>}>
        body
      </Panel>,
    );
    const region = screen.getByRole("region", { name: "Logit Lens" });
    expect(region).toHaveTextContent("Logit Lens");
    expect(region.querySelector('[data-status="derived"]')).not.toBeNull();
    expect(screen.getByText("L14")).toBeInTheDocument();
    expect(screen.getByText("body")).toBeInTheDocument();
  });
});

describe("Button", () => {
  it("defaults to type=button and applies the variant", () => {
    const onClick = vi.fn();
    render(
      <Button variant="danger" onClick={onClick}>
        Ablate
      </Button>,
    );
    const btn = screen.getByRole("button", { name: "Ablate" });
    expect(btn).toHaveAttribute("type", "button");
    expect(btn.className).toContain("text-ns-danger");
    fireEvent.click(btn);
    expect(onClick).toHaveBeenCalledOnce();
  });
});

describe("Tabs", () => {
  function Harness() {
    const [v, setV] = useState<"lens" | "attn" | "dla">("lens");
    return (
      <Tabs
        aria-label="Views"
        value={v}
        onValueChange={setV}
        items={[
          { value: "lens", label: "Logit Lens", status: "derived" },
          { value: "attn", label: "Attention", status: "observed" },
          { value: "dla", label: "DLA", disabled: true },
        ]}
      />
    );
  }

  it("marks the selected tab and moves selection with arrow keys, skipping disabled tabs", () => {
    render(<Harness />);
    const lens = screen.getByRole("tab", { name: "Logit Lens" });
    const attn = screen.getByRole("tab", { name: "Attention" });
    expect(lens).toHaveAttribute("aria-selected", "true");
    expect(attn).toHaveAttribute("tabindex", "-1");

    fireEvent.keyDown(lens, { key: "ArrowRight" });
    expect(attn).toHaveAttribute("aria-selected", "true");
    expect(attn).toHaveFocus();

    // wraps around past the disabled "DLA" tab
    fireEvent.keyDown(attn, { key: "ArrowRight" });
    expect(lens).toHaveAttribute("aria-selected", "true");

    fireEvent.click(attn);
    expect(attn).toHaveAttribute("aria-selected", "true");
  });
});
