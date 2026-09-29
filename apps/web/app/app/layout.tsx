import type { Metadata } from "next";
import { WorkspaceShell } from "@/components/workspace/WorkspaceShell";
import "./workspace.css";

export const metadata: Metadata = {
  title: "Neural Debugger",
  description: "NeuralScope research workspace: token × layer explorer, Logit Lens, attention, causal interventions and Math Lab.",
};

/**
 * The workspace shell (provider, header, demo banner, model gate) lives in the layout
 * so its state survives navigation between /app and /app/math.
 */
export default function WorkspaceLayout({ children }: LayoutProps<"/app">) {
  return <WorkspaceShell>{children}</WorkspaceShell>;
}
