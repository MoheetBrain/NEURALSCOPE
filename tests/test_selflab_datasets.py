"""SA1 datasets: Experiment A source corpus and Experiment B carry items.

Fast: uses only the cached Qwen2.5 tokenizer (no model weights). Tokenizer-dependent tests skip if it is not cached.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import pytest

from neuralscope.selflab.datasets import carry_items as ci
from neuralscope.selflab.datasets import source_corpus as sc
from neuralscope.selflab.provenance import sha256_text
from neuralscope.selflab.schemas import SOURCE_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    except OSError as e:  # not cached on this machine
        pytest.skip(f"tokenizer {TOKENIZER_ID} not cached: {e}")


def _all_renders(items):
    for it in items:
        for fam in sc.FAMILIES:
            for cond in sc.CONDITIONS:
                yield it, fam, cond, sc.render_turn1(it, cond, fam)


# ------------------------------------------------------------------------------------------ public API shape
def test_public_constants():
    assert sc.ANSWER_ABSTAIN == "insufficient evidence"
    assert sc.CONDITIONS == ("doc_v", "doc_alt", "no_doc", "doc_v_hint_conflict", "no_doc_hint", "doc_v_hint_agree")
    assert len(sc.FAMILIES) >= 4 and all(k == f.id for k, f in sc.FAMILIES.items())
    assert set(sc.DEFAULT_HELDOUT_FAMILIES) <= set(sc.FAMILIES)
    disc = set(sc.families_for_split("discovery"))
    held = set(sc.families_for_split("heldout_confirmation"))
    assert disc and held and not (disc & held) and held == set(sc.DEFAULT_HELDOUT_FAMILIES)
    assert set(sc.families_for_split("debug_pilot")) == disc
    for word in ("condition", "control", "experiment", "test", "source", "document"):
        assert word not in sc.SYSTEM_PROMPT.lower()


# ------------------------------------------------------------------------------------------ items
def test_determinism_and_prefix_stability():
    a, b = sc.make_items(30, 7), sc.make_items(30, 7)
    assert a == b
    assert sc.make_items(10, 7) == a[:10]
    assert sc.make_items(30, 8) != a
    assert len({it.item_id for it in a}) == 30


def test_values_distinct_and_in_set():
    items = sc.make_items(120, 3)
    assert len({it.entity for it in items}) == len(items)
    for it in items:
        assert len({it.v, it.v_alt, it.w}) == 3
        assert {it.v, it.v_alt, it.w} <= set(it.value_set)
        assert it.value_set == sc.ATTRIBUTES[it.attribute].values
        assert 1 <= len(it.distractors) <= 2 and 0 <= it.target_index <= len(it.distractors)
        for attr, val in it.distractors:
            assert attr != it.attribute and val in sc.ATTRIBUTES[attr].values
            assert val not in (it.v, it.v_alt, it.w)
        low = it.entity.lower()
        assert not any(s in low for s in sc.LEAK_SUBSTRINGS)
        assert not any(v.lower() in low for a in sc.ATTRIBUTES.values() for v in a.values)
    # attributes are balanced in blocks
    counts = {a: sum(it.attribute == a for it in items[: 7 * len(sc.ATTRIBUTES)]) for a in sc.ATTRIBUTES}
    assert set(counts.values()) == {7}


def test_value_sets_disjoint():
    seen: dict[str, str] = {}
    for a in sc.ATTRIBUTES.values():
        assert len(a.values) >= 3 and len(set(a.values)) == len(a.values)
        assert a.fact.count("{V}") == 1
        for v in a.values:
            assert v.lower() not in seen, (v, a.id, seen.get(v.lower()))
            seen[v.lower()] = a.id


def test_value_forms_are_single_tokens(tok):
    for a in sc.ATTRIBUTES.values():
        for v in a.values:
            for form in (" " + v, v, v[0].upper() + v[1:]):
                assert len(tok.encode(form, add_special_tokens=False)) == 1, (a.id, form)


def test_alignment_in_document_context(tok):
    for seed in (0, 1, 2):
        items, log = sc.make_items_with_log(60, seed, tok)
        assert log == []  # curated single-token sets: no redraw needed
        assert items == sc.make_items(60, seed)  # tokenizer does not change the draw when nothing is rejected
        for it in items:
            chk = sc.check_alignment(it, tok)
            assert chk["ok"], (it, chk)
            for rec in chk["per_family"].values():
                assert rec["n_tokens_v"] == rec["n_tokens_alt"]
                assert len(rec["diff_positions"]) == 1
            assert len(tok.encode(" " + it.entity, add_special_tokens=False)) >= 2


class _WordTok:
    """Toy tokenizer (no chat template): one id per word, +1 for capitalised words, +1 more for 'purple'."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids.append(zlib.crc32(w.encode()) % 50000 + 10)
            if w[:1].isupper():
                ids.append(1)
            if w.startswith("purple"):
                ids.append(2)
        return ids


def test_misaligned_draws_are_redrawn_and_logged():
    t = _WordTok()
    items, log = sc.make_items_with_log(150, 0, t)
    items2, log2 = sc.make_items_with_log(150, 0, t)
    assert (items, log) == (items2, log2)  # deterministic redraws
    mis = [e for e in log if e["reason"] == "value_misaligned"]
    assert mis, "expected at least one purple draw to be rejected"
    assert all("purple" in (e["v"], e["v_alt"]) for e in mis)
    assert all("purple" not in (it.v, it.v_alt) for it in items)
    assert all(sc.check_alignment(it, t)["ok"] for it in items)


# ------------------------------------------------------------------------------------------ rendering
def test_segments_slice_exactly_and_conditions_match():
    items = sc.make_items(8, 11)
    for it, fam, cond, (msgs, segs, meta) in _all_renders(items):
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == sc.SYSTEM_PROMPT
        content = msgs[1]["content"]
        by_kind = {s.kind: s for s in segs}
        assert len(by_kind) == len(segs)
        for s in segs:
            assert s.message_index == 1
            assert s.sha256 == sha256_text(content[s.char_start:s.char_end])
        assert content[by_kind["question"].char_start:by_kind["question"].char_end] == meta["question_text"]
        doc_key, hint_key = sc._CONDITION_TABLE[cond]
        assert meta["condition"] == cond and meta["family"] == fam
        assert meta["document_value"] == (getattr(it, doc_key) if doc_key else None)
        assert meta["hint_value"] == (getattr(it, hint_key) if hint_key else None)
        if doc_key:
            d = by_kind["document"]
            assert content[d.char_start:d.char_end] == meta["document_text"]
            assert re.search(rf"\b{re.escape(meta['document_value'])}\b", meta["document_text"])
        else:
            assert "document" not in by_kind and meta["document_text"] is None
        if hint_key:
            h = by_kind["user_hint"]
            assert content[h.char_start:h.char_end] == meta["hint_text"]
            assert re.search(rf"\b{re.escape(meta['hint_value'])}\b", meta["hint_text"])
        else:
            assert "user_hint" not in by_kind and meta["hint_text"] is None
        # values appear only where the condition puts them
        present = {x for x in (meta["document_value"], meta["hint_value"]) if x}
        for val in (it.v, it.v_alt, it.w):
            n = len(re.findall(rf"\b{re.escape(val)}\b", content))
            assert (n > 0) == (val in present), (cond, fam, val)
        assert f'"{sc.ANSWER_ABSTAIN}"' in content


def test_no_leakage_in_rendered_text():
    items = sc.make_items(10, 5)
    banned_re = [re.compile(re.escape(s), re.I) for s in sc.LEAK_SUBSTRINGS]
    banned_re += [re.compile(rf"\b{re.escape(c)}\b") for c in sc.CONDITIONS]
    banned_re += [re.compile(rf"\b{re.escape(f)}\b") for f in sc.FAMILIES]
    banned_re += [re.compile(r"\bF\d+\b"), re.compile(r"src-s\d+")]
    orders = [o for pair in sc.label_orders(0, 6) for o in pair]
    texts = []
    for it, fam, cond, (msgs, _, _) in _all_renders(items):
        texts += [(it, m["content"]) for m in msgs]
    for fam in sc.FAMILIES:
        texts += [(None, sc.render_source_question(o, fam)[0]) for o in orders]
    for it, text in texts:
        for rx in banned_re:
            assert not rx.search(text), (rx.pattern, text)
        if it is not None:
            assert it.item_id not in text


def test_label_orders_and_source_question():
    pairs = sc.label_orders(3, 50)
    assert len(pairs) == 50 and pairs == sc.label_orders(3, 50)
    assert len({tuple(p) for p, _ in pairs}) > 1
    for perm, rev in pairs:
        assert sorted(perm) == sorted(SOURCE_CATEGORIES) and rev == perm[::-1]
    for fam in sc.FAMILIES:
        for perm, rev in pairs[:5]:
            for order in (perm, rev):
                text, letter_map = sc.render_source_question(order, fam)
                assert list(letter_map) == ["A", "B", "C", "D"]
                assert list(letter_map.values()) == order
                assert sorted(letter_map.values()) == sorted(SOURCE_CATEGORIES)  # bijection
                lines = text.split("\n")
                assert lines[-1] == "Reply with a single letter."
                assert [ln[:3] for ln in lines[1:5]] == ["A. ", "B. ", "C. ", "D. "]
    with pytest.raises(ValueError):
        sc.render_source_question(["document", "document", "other_or_unknown", "insufficient_evidence"], "F1")


def test_render_turn2_appends_answer_verbatim():
    it = sc.make_items(1, 0)[0]
    msgs, _, _ = sc.render_turn1(it, "doc_v", "F1")
    order = list(SOURCE_CATEGORIES)
    m2, letter_map = sc.render_turn2(msgs, " East.\n", order, "F1")
    assert m2[:2] == msgs and m2[2] == {"role": "assistant", "content": " East.\n"}
    assert m2[3]["content"] == sc.render_source_question(order, "F1")[0]


def test_preset_case(tok):
    p = sc.preset_case(0)
    assert p == sc.preset_case(0)
    v, v_alt = p["expected_values"]["v"], p["expected_values"]["v_alt"]
    assert v != v_alt
    assert re.sub(rf"\b{re.escape(v)}\b", v_alt, p["document"]) == p["replacement_text"]
    assert "source" in p["question"].lower()
    n1 = len(tok.encode(p["document"], add_special_tokens=False))
    n2 = len(tok.encode(p["replacement_text"], add_special_tokens=False))
    assert n1 == n2


# ------------------------------------------------------------------------------------------ carry items
def test_carry_labels_and_template():
    probs = ci.make_carry_problems(40, 0)
    assert probs == ci.make_carry_problems(40, 0)
    assert sum(p["carry"] for p in probs) == 40 and len(probs) == 80
    for p in probs:
        a, b = p["a"], p["b"]
        assert 10 <= a <= 99 and 10 <= b <= 99 and p["sum"] == a + b < 100
        assert p["carry"] == ((a % 10) + (b % 10) >= 10)
        assert p["true_tens"] == (a + b) // 10
        expected_alt = a // 10 + b // 10 + (0 if p["carry"] else 1)
        assert p["alt_tens"] == expected_alt and p["alt_tens"] != p["true_tens"] and 0 <= p["alt_tens"] <= 9
        assert p["prompt"] == f"What is {a} + {b}? Give only the answer."


def test_carry_matches_recorded_e1_problems():
    path = ROOT / "docs/research/E1_carry_heads.json"
    if not path.exists():
        pytest.skip("E1 record not present")
    rows = json.loads(path.read_text())["rows"]
    mine = ci.make_carry_problems(40, 0)
    assert [(r["a"], r["b"], r["carry"]) for r in rows] == [(p["a"], p["b"], p["carry"]) for p in mine]


def test_carry_splits_disjoint_and_exclude_known():
    s = ci.carry_splits(40, 60, 1)
    assert s == ci.carry_splits(40, 60, 1)
    key = lambda p: (min(p["a"], p["b"]), max(p["a"], p["b"]))  # noqa: E731
    cal, held = {key(p) for p in s["calibration"]}, {key(p) for p in s["heldout"]}
    assert len(cal) == 80 and len(held) == 120 and not (cal & held)
    known = ci.e1_problem_pairs() | ci.e2_problem_pairs()
    assert len(known) > 80 and not ((cal | held) & known)
    for split, rows in s.items():
        assert all(r["split"] == split for r in rows)
        assert sum(r["carry"] for r in rows) == len(rows) // 2
