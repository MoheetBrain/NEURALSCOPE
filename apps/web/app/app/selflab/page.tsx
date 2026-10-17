import type { Metadata } from "next";
import { SelfLabView } from "@/components/selflab/SelfLabView";

export const metadata: Metadata = {
  title: "SelfLab",
  description:
    "SelfLab (research prototype): select a claim a model made about its own answer, run committed counterfactual experiments, and compare the claim with measured evidence.",
};

export default function SelfLabPage() {
  return <SelfLabView />;
}
