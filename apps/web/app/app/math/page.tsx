import type { Metadata } from "next";
import { MathLabView } from "@/components/workspace/mathlab/MathLabView";

export const metadata: Metadata = {
  title: "Math Lab",
};

export default function MathLabPage() {
  return <MathLabView />;
}
