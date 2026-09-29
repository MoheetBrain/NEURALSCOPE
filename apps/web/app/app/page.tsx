import type { Metadata } from "next";
import { DebuggerView } from "@/components/workspace/DebuggerView";

export const metadata: Metadata = {
  title: "Neural Debugger",
};

export default function WorkspacePage() {
  return <DebuggerView />;
}
