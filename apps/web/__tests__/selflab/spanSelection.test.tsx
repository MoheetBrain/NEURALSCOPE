import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SelectableResponse } from "@/components/selflab/SelectableResponse";
import { chunkText, clampSpan, rangeToSpan, resolveBoundary, selectionToSpan, showBlank, type CharSpan } from "@/components/selflab/spanSelection";

/** Build a container of `<span data-char-start>` chunks (plus optional decorations) for `text`. */
function mount(text: string, cuts: number[], opts: { decorate?: boolean } = {}): HTMLDivElement {
  const el = document.createElement("div");
  const bounds = [0, ...cuts, text.length];
  for (let i = 0; i < bounds.length - 1; i++) {
    if (opts.decorate) {
      // decorations carry no data-char-start and must contribute no characters
      const deco = document.createElement("i");
      deco.setAttribute("aria-hidden", "true");
      el.appendChild(deco);
    }
    const span = document.createElement("span");
    span.setAttribute("data-char-start", String(bounds[i]));
    span.textContent = text.slice(bounds[i], bounds[i + 1]);
    el.appendChild(span);
  }
  document.body.appendChild(el);
  return el;
}

function textNode(el: Element, chunkIndex: number): Text {
  return el.querySelectorAll("[data-char-start]")[chunkIndex].firstChild as Text;
}

function select(startNode: Node, startOffset: number, endNode: Node, endOffset: number, backwards = false): Selection {
  const sel = window.getSelection() as Selection;
  sel.removeAllRanges();
  if (backwards && typeof sel.setBaseAndExtent === "function") {
    sel.setBaseAndExtent(endNode, endOffset, startNode, startOffset);
  } else {
    const r = document.createRange();
    r.setStart(startNode, startOffset);
    r.setEnd(endNode, endOffset);
    sel.addRange(r);
  }
  return sel;
}

afterEach(() => {
  window.getSelection()?.removeAllRanges();
  document.body.innerHTML = "";
});

const TEXT = "The eggs are green. The eggs are green, says the document.";

describe("span selection offsets", () => {
  it("maps a selection inside one chunk to response offsets", () => {
    const el = mount(TEXT, []);
    const t = textNode(el, 0);
    const span = selectionToSpan(el, select(t, 4, t, 8), TEXT);
    expect(span).toEqual({ char_start: 4, char_end: 8, text: "eggs" });
  });

  it("resolves repeated substrings by position, not by search", () => {
    const el = mount(TEXT, []);
    const t = textNode(el, 0);
    const second = TEXT.indexOf("green", 15); // the second "green"
    const span = selectionToSpan(el, select(t, second, t, second + 5), TEXT);
    expect(span?.char_start).toBe(second);
    expect(span?.text).toBe("green");
    expect(second).not.toBe(TEXT.indexOf("green"));
  });

  it("handles a selection spanning several text nodes", () => {
    // chunks: [0,10) [10,25) [25,end)
    const el = mount(TEXT, [10, 25]);
    const span = selectionToSpan(el, select(textNode(el, 0), 4, textNode(el, 2), 5), TEXT);
    expect(span).toEqual({ char_start: 4, char_end: 30, text: TEXT.slice(4, 30) });
  });

  it("ignores decorations without data-char-start", () => {
    const el = mount(TEXT, [10, 25], { decorate: true });
    const span = selectionToSpan(el, select(textNode(el, 1), 2, textNode(el, 2), 3), TEXT, { trim: false });
    expect(span).toEqual({ char_start: 12, char_end: 28, text: TEXT.slice(12, 28) });
    // default trimming drops the leading space at offset 12
    expect(selectionToSpan(el, window.getSelection(), TEXT)?.char_start).toBe(13);
  });

  it("normalises a backwards (right-to-left) selection", () => {
    const el = mount(TEXT, [10, 25]);
    const sel = select(textNode(el, 0), 4, textNode(el, 1), 3, true);
    const span = selectionToSpan(el, sel, TEXT);
    expect(span?.char_start).toBe(4);
    expect(span?.char_end).toBe(12); // "eggs are" — the trailing space at 12 is trimmed
  });

  it("resolves element boundaries (container + child index)", () => {
    const el = mount(TEXT, [10, 25]);
    // boundary before the 2nd chunk element → its char start; after the last → text end
    expect(resolveBoundary(el, el, 1, TEXT.length)).toBe(10);
    expect(resolveBoundary(el, el, 3, TEXT.length)).toBe(TEXT.length);
    const r = document.createRange();
    r.setStart(el, 0);
    r.setEnd(el, 2);
    expect(rangeToSpan(el, r, TEXT, { trim: false })).toEqual({ char_start: 0, char_end: 25, text: TEXT.slice(0, 25) });
  });

  it("trims surrounding whitespace by default", () => {
    const el = mount(TEXT, []);
    const t = textNode(el, 0);
    // " eggs are " (with spaces both sides)
    const span = selectionToSpan(el, select(t, 3, t, 13), TEXT);
    expect(span?.text).toBe("eggs are");
    expect(span?.char_start).toBe(4);
    expect(span?.char_end).toBe(12);
  });

  it("clips a selection that starts outside the container and returns null when disjoint", () => {
    const before = document.createElement("p");
    before.textContent = "outside";
    document.body.appendChild(before);
    const el = mount(TEXT, [10]);
    const span = selectionToSpan(el, select(before.firstChild as Text, 2, textNode(el, 0), 3), TEXT);
    expect(span).toEqual({ char_start: 0, char_end: 3, text: "The" });

    const after = document.createElement("p");
    after.textContent = "elsewhere";
    document.body.appendChild(after);
    expect(selectionToSpan(el, select(after.firstChild as Text, 0, after.firstChild as Text, 4), TEXT)).toBeNull();
  });

  it("keeps a whitespace-only selection of a blank response (DEMO model answers with newlines)", () => {
    const blank = "\n\n\n\n";
    const el = mount(blank, []);
    const t = textNode(el, 0);
    expect(selectionToSpan(el, select(t, 1, t, 3), blank)).toEqual({ char_start: 1, char_end: 3, text: "\n\n" });
  });

  it("returns null for collapsed selections", () => {
    const el = mount(TEXT, []);
    const t = textNode(el, 0);
    expect(selectionToSpan(el, select(t, 5, t, 5), TEXT)).toBeNull();
    expect(selectionToSpan(el, null, TEXT)).toBeNull();
  });

  it("validates manual offsets and splits text around a highlight", () => {
    expect(clampSpan("abc def", 5, 1)).toEqual({ char_start: 1, char_end: 5, text: "bc d" });
    expect(clampSpan("abc", -4, 99)).toEqual({ char_start: 0, char_end: 3, text: "abc" });
    expect(clampSpan("abc", 2, 2)).toBeNull();
    // a whitespace-only span is kept rather than trimmed to nothing (the backend accepts and notes it)
    expect(clampSpan("\n\n\n", 0, 3)).toEqual({ char_start: 0, char_end: 3, text: "\n\n\n" });
    expect(clampSpan("a \n b", 1, 4)).toEqual({ char_start: 1, char_end: 4, text: " \n " });
    expect(showBlank("\n \n")).toBe("↵·↵");
    expect(showBlank("two\nlines")).toBe("two↵lines");
    const chunks = chunkText("hello world", { char_start: 6, char_end: 11 });
    expect(chunks).toEqual([
      { start: 0, text: "hello ", marked: false },
      { start: 6, text: "world", marked: true },
    ]);
    expect(chunks.map((c) => c.text).join("")).toBe("hello world");
  });
});

describe("<SelectableResponse>", () => {
  function Harness({ text, onSpan }: { text: string; onSpan: (s: CharSpan | null) => void }) {
    const [v, setV] = useState<CharSpan | null>(null);
    return (
      <SelectableResponse
        text={text}
        value={v}
        claim={{ char_start: 0, char_end: 3 }}
        onChange={(s) => {
          setV(s);
          onSpan(s);
        }}
      />
    );
  }

  it("reads a mouse selection across the claim highlight and the plain text", () => {
    const onSpan = vi.fn();
    render(<Harness text={TEXT} onSpan={onSpan} />);
    const box = screen.getByTestId("selectable-response");
    // rendered verbatim: DOM text equals the response
    expect(box.textContent).toBe(TEXT);
    const chunks = box.querySelectorAll("[data-char-start]");
    expect(chunks).toHaveLength(2); // claim [0,3) + rest
    select(chunks[0].firstChild as Text, 1, chunks[1].firstChild as Text, 5);
    fireEvent.mouseUp(box);
    expect(onSpan).toHaveBeenLastCalledWith({ char_start: 1, char_end: 8, text: TEXT.slice(1, 8) });
    expect(screen.getByTestId("selection-readout")).toHaveTextContent("chars [1, 8)");
  });

  it("supports whole-response and manual offsets without a mouse", () => {
    const onSpan = vi.fn();
    render(<Harness text="  answer: green  " onSpan={onSpan} />);
    fireEvent.click(screen.getByRole("button", { name: /whole response/i }));
    expect(onSpan).toHaveBeenLastCalledWith({ char_start: 2, char_end: 15, text: "answer: green" });
    fireEvent.change(screen.getByLabelText("Claim start offset"), { target: { value: "10" } });
    expect(onSpan).toHaveBeenLastCalledWith({ char_start: 10, char_end: 15, text: "green" });
  });
});
