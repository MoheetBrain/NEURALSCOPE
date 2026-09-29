"use client";

import { Inspector } from "./Inspector";
import { MatrixPanel } from "./MatrixPanel";
import { PromptColumn } from "./PromptColumn";
import { Timeline } from "./Timeline";

/** /app — prompt & output (left) · token × depth matrix (centre) · inspector (right) · timeline (bottom). */
export function DebuggerView() {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid min-h-0 flex-1 grid-cols-[300px_minmax(0,1fr)_minmax(420px,34vw)] gap-2 px-2 pt-2">
        <div className="min-h-0 overflow-hidden">
          <PromptColumn />
        </div>
        <div className="min-h-0 pb-2">
          <MatrixPanel />
        </div>
        <div className="min-h-0 pb-2">
          <Inspector />
        </div>
      </div>
      <Timeline />
    </div>
  );
}
