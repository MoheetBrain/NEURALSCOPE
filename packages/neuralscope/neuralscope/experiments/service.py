"""NeuralScope service layer — orchestrates model runs, captures, interventions, comparisons.

Everything the API exposes goes through ``NeuralScopeService``. It owns:
  * the model adapter (instrumentation) and the experiment store (persistence);
  * compilation of user-facing intervention specs into adapter edits;
  * summarised, browser-sized views of server-side tensors;
  * the epistemic status attached to every quantity it returns.

Epistemic statuses (see docs/INTERPRETABILITY.md):
  OBSERVED  direct tensor / model measurement
  DERIVED   mathematical transformation of observed quantities
  INFERRED  semantic interpretation / hypothesis (never produced automatically in V1)
  CAUSAL    CAUSALLY TESTED — a measured effect of an actual intervention, with its scope
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import torch

from neuralscope.config import Settings
from neuralscope.experiments.store import ExperimentStore, new_id, now_iso
from neuralscope.instrumentation.adapter import (
    Edit,
    HeadScaleEdit,
    MLPScaleEdit,
    ModelAdapter,
    ResidAddEdit,
    ResidSetEdit,
    Trace,
    TraceLevel,
)
from neuralscope.interpretability.attribution import direct_logit_attribution, head_vocab_projection
from neuralscope.interpretability.lens import lens_at, lens_summary, lens_target
from neuralscope.interpretability.metrics import (
    cosine,
    entropy,
    kl_divergence,
    relative_delta,
    successive_cosine_change,
)
from neuralscope.schemas.interventions import (
    HeadAblation,
    MLPAblation,
    ResidualPatch,
    ResidualSteer,
    describe_all,
)

OBSERVED, DERIVED, INFERRED, CAUSAL = "observed", "derived", "inferred", "causal"
LEVEL_ORDER = {"minimal": 0, "standard": 1, "full": 2}
LENS_K = 10
FINAL_K = 20

Progress = Callable[[float, str], None]


class ServiceError(ValueError):
    """User-facing error (bad request)."""


def _r(x: float, nd: int = 5) -> float | None:
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def _rl(t: torch.Tensor, nd: int = 4) -> list:
    return [[_r(v, nd) for v in row] for row in t.tolist()] if t.dim() == 2 else [_r(v, nd) for v in t.tolist()]


class NeuralScopeService:
    def __init__(self, adapter: ModelAdapter, store: ExperimentStore, settings: Settings):
        self.adapter = adapter
        self.store = store
        self.settings = settings
        self._lens_target_cache: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self._run_locks: dict[str, threading.Lock] = {}
        self._run_locks_guard = threading.Lock()

    def _run_lock(self, run_id: str) -> threading.Lock:
        """Serialises lazy tensor upgrades of one run (concurrent UI requests must not double-write a file)."""
        with self._run_locks_guard:
            return self._run_locks.setdefault(run_id, threading.Lock())

    # ================================================================== helpers
    @property
    def info(self):
        return self.adapter.info

    def depth_labels(self) -> list[str]:
        return ["emb"] + [f"L{l}" for l in range(self.info.n_layers)]

    def _get_run(self, run_id: str) -> dict[str, Any]:
        rec = self.store.get_run(run_id)
        if rec is None:
            raise KeyError(f"run {run_id} not found")
        return rec

    def token_view(self, ids: list[int], prompt_len: int, user_span: list[int] | tuple | None, chat: bool) -> list[dict]:
        strs = self.adapter.token_strings(ids)
        out = []
        for i, (tid, s) in enumerate(zip(ids, strs)):
            if i >= prompt_len:
                role = "generated"
            elif user_span and user_span[0] <= i < user_span[1]:
                role = "user"
            elif not chat:
                role = "user"
            else:
                role = "template"
            out.append({"index": i, "id": int(tid), "text": s, "special": self.adapter.is_special(tid), "role": role})
        return out

    def _tok_str(self, tid: int) -> str:
        return self.adapter.token_strings([int(tid)])[0]

    # ================================================================== trace → tensors
    def _lens_tensors(self, resid: torch.Tensor) -> dict[str, torch.Tensor]:
        lens = lens_summary(self.adapter, resid, k=LENS_K)
        return {"lens_top_ids": lens.top_ids.to(torch.int32), "lens_top_probs": lens.top_probs,
                "lens_top_logits": lens.top_logits, "lens_entropy": lens.entropy}

    def ensure_lens(self, run_id: str) -> dict[str, torch.Tensor]:
        """Logit-lens summaries are computed eagerly for interactive runs and lazily for Math Lab runs."""
        with self._run_lock(run_id):
            t = self.store.load_tensors(run_id)
            if "lens_top_ids" not in t:
                self.store.add_tensors(run_id, self._lens_tensors(t["resid"]))
                t = self.store.load_tensors(run_id)
            return t

    def _trace_tensors(self, trace: Trace, prompt_len: int, with_lens: bool = True) -> dict[str, torch.Tensor]:
        T = trace.seq_len
        lp = torch.log_softmax(trace.logits, dim=-1)
        top = torch.topk(trace.logits, FINAL_K, dim=-1)
        nxt = torch.full((T,), float("nan"))
        nrank = torch.zeros(T, dtype=torch.long)
        ids = trace.input_ids
        if T > 1:
            nxt[:-1] = lp[torch.arange(T - 1), ids[1:]]
            tl = trace.logits[torch.arange(T - 1), ids[1:]]
            nrank[:-1] = 1 + (trace.logits[:-1] > tl[:, None]).sum(-1)
        start = max(prompt_len - 1, 0)
        t = {
            "input_ids": ids,
            "resid": trace.resid.float(),
            "final_top_ids": top.indices,
            "final_top_logprobs": lp.gather(-1, top.indices),
            "final_entropy": entropy(lp),
            "next_logprob": nxt,
            "next_rank": nrank,
            "gen_logprobs": lp[start:].float(),
        }
        if with_lens:
            t.update(self._lens_tensors(trace.resid))
        if trace.attn is not None:
            t["attn"] = trace.attn.half()
        if trace.head_z is not None:
            t["head_z"] = trace.head_z.float()
        if trace.mlp_out is not None:
            t["mlp_out"] = trace.mlp_out.float()
        return t

    def _trace_from_tensors(self, t: dict[str, torch.Tensor], level: str) -> Trace:
        """Rebuild a Trace-like object from stored tensors (final logits are not stored in full)."""
        tr = Trace(input_ids=t["input_ids"], resid=t["resid"], logits=torch.empty(0), level=level)  # type: ignore[arg-type]
        tr.attn = t["attn"].float() if "attn" in t else None
        tr.head_z = t.get("head_z")
        tr.mlp_out = t.get("mlp_out")
        return tr

    # ================================================================== compile interventions
    def compile(self, specs: list, n_seq: int | None = None) -> list[Edit]:
        info = self.info
        edits: list[Edit] = []
        for s in specs:
            if isinstance(s, dict):
                from pydantic import TypeAdapter

                from neuralscope.schemas.interventions import Intervention

                s = TypeAdapter(Intervention).validate_python(s)
            if isinstance(s, HeadAblation):
                if not (0 <= s.layer < info.n_layers and 0 <= s.head < info.n_heads):
                    raise ServiceError(f"head L{s.layer}.H{s.head} out of range ({info.n_layers} layers × {info.n_heads} heads)")
                edits.append(HeadScaleEdit(s.layer, s.head, s.factor, s.positions))
            elif isinstance(s, MLPAblation):
                if not 0 <= s.layer < info.n_layers:
                    raise ServiceError(f"MLP layer {s.layer} out of range")
                edits.append(MLPScaleEdit(s.layer, s.factor, s.positions))
            elif isinstance(s, ResidualPatch):
                if not 0 <= s.depth <= info.n_layers:
                    raise ServiceError(f"depth {s.depth} out of range 0..{info.n_layers}")
                src = self.store.load_tensors(s.source_run_id)["resid"]
                sp = s.position if s.source_position is None else s.source_position
                if not 0 <= sp < src.shape[1]:
                    raise ServiceError(f"source position {sp} out of range for run {s.source_run_id}")
                if n_seq is not None and not 0 <= s.position < n_seq:
                    raise ServiceError(f"patch position {s.position} out of range")
                edits.append(ResidSetEdit(s.depth, s.position, src[s.depth, sp].clone(), s.alpha))
            elif isinstance(s, ResidualSteer):
                a = self.store.load_tensors(s.plus_run_id)["resid"][s.depth, s.plus_position]
                b = self.store.load_tensors(s.minus_run_id)["resid"][s.depth, s.minus_position]
                edits.append(ResidAddEdit(s.depth, (a - b) * s.scale, s.positions))
            else:
                raise ServiceError(f"unsupported intervention {s!r}")
        return edits

    @staticmethod
    def _edits_from_depth(edits: list[Edit], start_depth: int) -> list[Edit]:
        """Edits still to be applied when resuming from a stored residual at ``start_depth``.

        The stored residual was captured *with* the run's edits, so edits acting before
        ``start_depth`` — and residual edits exactly at it — are already baked in.
        """
        keep: list[Edit] = []
        for e in edits:
            if isinstance(e, (HeadScaleEdit, MLPScaleEdit)):
                if e.layer >= start_depth:
                    keep.append(e)
            elif e.depth > start_depth:
                keep.append(e)
        return keep

    @staticmethod
    def _dump_specs(specs: list) -> list[dict]:
        return [s if isinstance(s, dict) else s.model_dump() for s in specs]

    # ================================================================== targets
    def resolve_target(self, input_ids: list[int], prompt_len: int, target_text: str | None = None,
                       target_token_id: int | None = None, target_position: int | None = None,
                       contrast_text: str | None = None, contrast_token_id: int | None = None) -> dict:
        T = len(input_ids)
        pos = prompt_len - 1 if target_position is None else int(target_position)
        if not 0 <= pos < T:
            raise ServiceError(f"target position {pos} outside sequence of length {T}")
        note = None
        if target_token_id is not None:
            tid, source = int(target_token_id), "token_id"
        elif target_text:
            enc = self.adapter.encode(target_text)
            if not enc:
                raise ServiceError("target text tokenises to nothing")
            tid, source = enc[0], "text"
            if len(enc) > 1:
                note = (f"'{target_text}' tokenises to {len(enc)} tokens {self.adapter.token_strings(enc)}; "
                        f"tracking the first token at this position.")
        elif pos + 1 < T:
            tid, source = int(input_ids[pos + 1]), "next_token"
        else:
            tid, source = None, "argmax"
        cid = None
        if contrast_token_id is not None:
            cid = int(contrast_token_id)
        elif contrast_text:
            enc = self.adapter.encode(contrast_text)
            cid = enc[0] if enc else None
        return {
            "position": pos, "token_id": tid, "token": self._tok_str(tid) if tid is not None else None,
            "contrast_id": cid, "contrast": self._tok_str(cid) if cid is not None else None,
            "source": source, "note": note,
        }

    # ================================================================== runs
    def create_run(self, prompt: str, chat: bool = True, system_prompt: str | None = None,
                   max_new_tokens: int = 16, trace_level: TraceLevel | None = None,
                   target_text: str | None = None, target_token_id: int | None = None,
                   target_position: int | None = None, contrast_text: str | None = None,
                   notes: str | None = None, kind: str = "baseline", math: dict | None = None,
                   benchmark_id: str | None = None, tags: list[str] | None = None) -> dict:
        if not prompt or not prompt.strip():
            raise ServiceError("prompt is empty")
        level = trace_level or self.settings.trace_level
        if level not in LEVEL_ORDER:
            raise ServiceError(f"unknown trace level {level}")
        max_new_tokens = max(1, min(int(max_new_tokens), self.settings.max_new_tokens_cap))
        tp = self.adapter.tokenize(prompt, chat=chat, system_prompt=system_prompt)
        if len(tp.input_ids) > self.settings.max_prompt_tokens:
            raise ServiceError(f"prompt is {len(tp.input_ids)} tokens; V1 limit is {self.settings.max_prompt_tokens} "
                               "(activation storage grows with T² for attention)")
        t0 = time.time()
        gen = self.adapter.generate(tp.input_ids, max_new_tokens)
        t_gen = time.time() - t0
        ids = tp.input_ids + gen.generated_ids
        t1 = time.time()
        trace = self.adapter.capture(ids, level)
        t_cap = time.time() - t1
        P = len(tp.input_ids)
        # integrity check: the teacher-forced capture must reproduce every greedy choice
        consistent = all(int(trace.logits[P - 1 + i].argmax()) == g for i, g in enumerate(gen.generated_ids))
        tensors = self._trace_tensors(trace, P, with_lens=False)  # lens summaries computed on first view
        target = self.resolve_target(ids, P, target_text, target_token_id, target_position, contrast_text)
        rec = {
            "id": new_id("run"),
            "created_at": now_iso(),
            "kind": kind,
            "model_id": self.info.model_id,
            "is_demo": self.info.is_demo,
            "prompt": prompt,
            "prompt_text": tp.text,
            "chat": chat,
            "system_prompt": system_prompt,
            "generation": {"decoding": "greedy", "max_new_tokens": max_new_tokens, "stopped_by": gen.stopped_by,
                           "kv_cache": True, "attn_implementation": "eager", "dtype": self.info.dtype,
                           "seconds_generate": round(t_gen, 3), "seconds_capture": round(t_cap, 3)},
            "input_ids": ids,
            "prompt_len": P,
            "user_span": list(tp.user_span) if tp.user_span else None,
            "generated_ids": gen.generated_ids,
            "generated_text": self.adapter.decode(gen.generated_ids),
            "target": target,
            "trace_level": level,
            "parent_run_id": None,
            "interventions": [],
            "metrics": {},
            "math": math,
            "correctness": (math or {}).get("verification", {}).get("status") if math else None,
            "notes": notes,
            "tags": tags or [],
            "benchmark_id": benchmark_id,
            "trace_consistent": consistent,
        }
        return self.store.save_run(rec, tensors)

    def ensure_level(self, run_id: str, level: TraceLevel) -> dict[str, torch.Tensor]:
        """Lazily upgrade a run's stored trace (deterministic re-capture) to at least ``level``."""
        with self._run_lock(run_id):
            return self._ensure_level_locked(run_id, level)

    def _ensure_level_locked(self, run_id: str, level: TraceLevel) -> dict[str, torch.Tensor]:
        rec = self._get_run(run_id)
        t = self.store.load_tensors(run_id)
        if LEVEL_ORDER[rec["trace_level"]] >= LEVEL_ORDER[level]:
            return t
        edits = self.compile(rec.get("interventions") or [], n_seq=len(rec["input_ids"]))
        trace = self.adapter.capture(rec["input_ids"], level, edits)
        drift = float((trace.resid - t["resid"]).abs().max())
        extra: dict[str, torch.Tensor] = {}
        if trace.attn is not None:
            extra["attn"] = trace.attn.half()
        if trace.head_z is not None:
            extra["head_z"] = trace.head_z
        if trace.mlp_out is not None:
            extra["mlp_out"] = trace.mlp_out
        self.store.add_tensors(run_id, extra)
        m = dict(rec.get("metrics") or {})
        m["lazy_recapture"] = {"level": level, "max_abs_resid_drift": drift, "at": now_iso()}
        self.store.update_run(run_id, trace_level=level, metrics=m)
        return self.store.load_tensors(run_id)

    def run_summary(self, rec: dict) -> dict:
        v = rec.get("math") or {}
        return {
            "id": rec["id"], "created_at": rec["created_at"], "kind": rec["kind"], "model_id": rec["model_id"],
            "is_demo": rec["is_demo"], "prompt": rec["prompt"], "generated_text": rec["generated_text"],
            "free_generated_text": rec.get("free_generated_text"), "parent_run_id": rec.get("parent_run_id"),
            "interventions_label": describe_all(self._parse_specs(rec.get("interventions") or [])),
            "n_tokens": len(rec["input_ids"]), "trace_level": rec["trace_level"],
            "correctness": rec.get("correctness"), "benchmark_id": rec.get("benchmark_id"),
            "notes": rec.get("notes"), "problem_kind": (v.get("problem") or {}).get("kind"),
        }

    @staticmethod
    def _parse_specs(specs: list[dict]) -> list:
        from pydantic import TypeAdapter

        from neuralscope.schemas.interventions import Intervention

        ta = TypeAdapter(Intervention)
        return [ta.validate_python(s) for s in specs]

    def run_detail(self, run_id: str) -> dict:
        rec = self._get_run(run_id)
        t = self.store.load_tensors(run_id)
        ids = rec["input_ids"]
        tokens = self.token_view(ids, rec["prompt_len"], rec.get("user_span"), rec["chat"])
        top_ids, top_lp = t["final_top_ids"], t["final_top_logprobs"]
        for i, tok in enumerate(tokens):
            tok["next_logprob"] = _r(t["next_logprob"][i])
            tok["next_rank"] = int(t["next_rank"][i]) if i < len(ids) - 1 else None
            tok["entropy"] = _r(t["final_entropy"][i], 4)
            tok["top"] = [{"id": int(a), "text": self._tok_str(a), "logprob": _r(b)}
                          for a, b in zip(top_ids[i, :5].tolist(), top_lp[i, :5].tolist())]
        out = dict(rec)
        out["tokens"] = tokens
        out["depth_labels"] = self.depth_labels()
        out["interventions_label"] = describe_all(self._parse_specs(rec.get("interventions") or []))
        out["children"] = [self.run_summary(c) for c in self.store.list_runs(parent_run_id=run_id, limit=50)]
        out["status"] = {"tokens": OBSERVED, "generation": OBSERVED, "top_predictions": OBSERVED}
        return out

    # ================================================================== matrix
    MATRIX_METRICS = {
        "lens_top1_prob": ("Logit Lens top-1 probability", DERIVED, "linear", "sequential"),
        "lens_entropy": ("Logit Lens entropy (nats)", DERIVED, "linear", "sequential_r"),
        "target_prob": ("Logit Lens P(target)", DERIVED, "linear", "sequential"),
        "target_rank": ("Logit Lens rank of target (log₁₀)", DERIVED, "log", "sequential_r"),
        "next_token_rank": ("Logit Lens rank of actual next token (log₁₀)", DERIVED, "log", "sequential_r"),
        "resid_norm": ("Residual-stream L2 norm", DERIVED, "log", "sequential"),
        "cos_delta": ("1 − cos(h_d, h_{d−1}) — rotation by each block", DERIVED, "linear", "sequential"),
        "delta_resid": ("‖h_int − h_base‖ / ‖h_base‖ (intervention effect)", CAUSAL, "linear", "sequential"),
    }

    def _lens_target_all(self, run_id: str, token_id: int) -> dict[str, torch.Tensor]:
        key = (run_id, int(token_id))
        if key not in self._lens_target_cache:
            t = self.store.load_tensors(run_id)
            lt = lens_target(self.adapter, t["resid"], int(token_id))
            self._lens_target_cache[key] = {"prob": lt.prob, "rank": lt.rank, "logit": lt.logit}
            if len(self._lens_target_cache) > 16:
                self._lens_target_cache.pop(next(iter(self._lens_target_cache)))
        return self._lens_target_cache[key]

    def matrix(self, run_id: str, metric: str = "lens_top1_prob", target_token_id: int | None = None) -> dict:
        rec = self._get_run(run_id)
        if metric not in self.MATRIX_METRICS:
            raise ServiceError(f"unknown metric {metric}; choose from {list(self.MATRIX_METRICS)}")
        t = self.ensure_lens(run_id)
        label, status, scale, cmap = self.MATRIX_METRICS[metric]
        D, T = t["resid"].shape[0], t["resid"].shape[1]
        extra: dict[str, Any] = {}
        if metric == "lens_top1_prob":
            vals = t["lens_top_probs"][:, :, 0]
            extra["top1_tokens"] = [[self._tok_str(i) for i in row] for row in t["lens_top_ids"][:, :, 0].tolist()]
        elif metric == "lens_entropy":
            vals = t["lens_entropy"]
        elif metric in ("target_prob", "target_rank"):
            tid = target_token_id if target_token_id is not None else (rec.get("target") or {}).get("token_id")
            if tid is None:
                raise ServiceError("no target token for this run")
            lt = self._lens_target_all(run_id, tid)
            vals = lt["prob"] if metric == "target_prob" else torch.log10(lt["rank"].float())
            extra["target"] = {"token_id": int(tid), "token": self._tok_str(tid)}
            if metric == "target_rank":
                extra["ranks"] = lt["rank"].tolist()
        elif metric == "next_token_rank":
            ids = t["input_ids"]
            ranks = torch.ones(D, T)
            for d in range(D):
                lg = self.adapter.logit_lens(t["resid"][d, :-1])
                tl = lg[torch.arange(T - 1), ids[1:]]
                ranks[d, :-1] = 1 + (lg > tl[:, None]).sum(-1).float()
            ranks[:, -1] = float("nan")
            vals = torch.log10(ranks)
        elif metric == "resid_norm":
            vals = t["resid"].norm(dim=-1)
        elif metric == "cos_delta":
            vals = torch.cat([torch.zeros(1, T), successive_cosine_change(t["resid"])])
        elif metric == "delta_resid":
            parent = rec.get("parent_run_id")
            if not parent:
                raise ServiceError("delta_resid requires an intervention run (it compares against its parent)")
            base = self.store.load_tensors(parent)["resid"]
            vals = relative_delta(base, t["resid"])
            extra["parent_run_id"] = parent
        finite = vals[torch.isfinite(vals)]
        return {
            "run_id": run_id, "metric": metric, "label": label, "status": status, "scale": scale, "colormap": cmap,
            "depth_labels": self.depth_labels(), "n_depths": D, "n_tokens": T,
            "values": _rl(vals.float(), 4),
            "vmin": _r(finite.min()) if finite.numel() else 0.0, "vmax": _r(finite.max()) if finite.numel() else 1.0,
            **extra,
        }

    # ================================================================== cell
    def cell(self, run_id: str, depth: int, position: int, target_token_id: int | None = None) -> dict:
        rec = self._get_run(run_id)
        t = self.store.load_tensors(run_id)
        R = t["resid"]
        D, T, d = R.shape
        if not (0 <= depth < D and 0 <= position < T):
            raise ServiceError(f"cell ({depth},{position}) outside {D}×{T}")
        h = R[depth, position]
        lens = lens_at(self.adapter, h, k=LENS_K)
        ids = rec["input_ids"]
        next_id = ids[position + 1] if position + 1 < T else None
        tid = target_token_id if target_token_id is not None else (rec.get("target") or {}).get("token_id")
        lg = lens["logits_full"]
        absval = h.abs()
        topd = torch.topk(absval, 12)
        hist = torch.histc(h, bins=40, min=float(h.min()), max=float(h.max()))
        info: dict[str, Any] = {
            "run_id": run_id, "depth": depth, "depth_label": self.depth_labels()[depth], "position": position,
            "token": self._tok_str(ids[position]), "token_id": ids[position],
            "layer_note": ("token embedding (input to block 0)" if depth == 0 else
                           f"residual stream after block {depth - 1} (pre final-norm)"),
            "vector": {
                "status": OBSERVED, "d_model": d,
                "values": [_r(v, 4) for v in h.tolist()],
                "norm": _r(h.norm()), "mean": _r(h.mean()), "std": _r(h.std()), "max_abs": _r(absval.max()),
                "top_dims": [{"dim": int(i), "value": _r(h[i])} for i in topd.indices.tolist()],
                "histogram": {"counts": [int(c) for c in hist.tolist()], "min": _r(h.min()), "max": _r(h.max())},
            },
            "geometry": {
                "status": DERIVED,
                "cos_prev_depth": _r(cosine(h, R[depth - 1, position])) if depth > 0 else None,
                "cos_next_depth": _r(cosine(h, R[depth + 1, position])) if depth + 1 < D else None,
                "cos_final_depth": _r(cosine(h, R[-1, position])),
                "norm_ratio_prev": _r(h.norm() / R[depth - 1, position].norm()) if depth > 0 else None,
            },
            "lens": {
                "status": DERIVED,
                "top": [{"id": int(i), "text": self._tok_str(i), "logit": _r(a), "prob": _r(p)}
                        for i, a, p in zip(lens["ids"], lens["logits"], lens["probs"])],
                "entropy": _r(lens["entropy"]),
            },
        }
        if next_id is not None:
            info["next_token"] = {"id": int(next_id), "text": self._tok_str(next_id),
                                  "lens_rank": int(1 + (lg > lg[next_id]).sum()),
                                  "lens_prob": _r(lens["logprobs_full"][next_id].exp())}
        if tid is not None:
            info["target"] = {"id": int(tid), "text": self._tok_str(tid), "lens_rank": int(1 + (lg > lg[tid]).sum()),
                              "lens_prob": _r(lens["logprobs_full"][tid].exp()), "lens_logit": _r(lg[tid])}
        return info

    # ================================================================== logit lens through depth
    def logit_lens(self, run_id: str, position: int | None = None, target_token_id: int | None = None,
                   k: int = LENS_K) -> dict:
        rec = self._get_run(run_id)
        t = self.store.load_tensors(run_id)
        R = t["resid"]
        D, T, _ = R.shape
        tgt = rec.get("target") or {}
        pos = tgt.get("position", rec["prompt_len"] - 1) if position is None else int(position)
        if not 0 <= pos < T:
            raise ServiceError("position out of range")
        tid = target_token_id if target_token_id is not None else tgt.get("token_id")
        lg = self.adapter.logit_lens(R[:, pos])  # [D, V]
        lp = torch.log_softmax(lg, -1)
        top = torch.topk(lg, k, -1)
        rows = []
        for dd in range(D):
            row = {"depth": dd, "label": self.depth_labels()[dd],
                   "top": [{"id": int(i), "text": self._tok_str(i), "logit": _r(a), "prob": _r(lp[dd, i].exp())}
                           for i, a in zip(top.indices[dd].tolist(), top.values[dd].tolist())],
                   "entropy": _r(-(lp[dd].exp() * lp[dd]).sum())}
            if tid is not None:
                row["target"] = {"rank": int(1 + (lg[dd] > lg[dd, tid]).sum()), "prob": _r(lp[dd, tid].exp()),
                                 "logit": _r(lg[dd, tid]), "logprob": _r(lp[dd, tid])}
            rows.append(row)
        first_top1 = None
        if tid is not None:
            for r in rows:
                if r["target"]["rank"] == 1:
                    first_top1 = r["depth"]
                    break
        ids = rec["input_ids"]
        return {
            "run_id": run_id, "position": pos, "token": self._tok_str(ids[pos]),
            "predicts_next": self._tok_str(ids[pos + 1]) if pos + 1 < T else None,
            "target": {"id": tid, "text": self._tok_str(tid)} if tid is not None else None,
            "depths": rows, "first_depth_target_top1": first_top1, "status": DERIVED,
            "method": "logits = W_U · RMSNorm_final(h_d) using the model's own final norm (γ, per-vector rms) and "
                      "unembedding; exact at the final depth, biased at early depths (not a tuned lens).",
        }

    # ================================================================== attention
    def attention(self, run_id: str, layer: int, query_position: int | None = None) -> dict:
        rec = self._get_run(run_id)
        t = self.ensure_level(run_id, "standard")
        A = t["attn"].float()  # [N, H, T, T]
        N, H, T, _ = A.shape
        if not 0 <= layer < N:
            raise ServiceError(f"layer {layer} out of range 0..{N - 1}")
        q = (rec.get("target") or {}).get("position", rec["prompt_len"] - 1) if query_position is None else int(query_position)
        if not 0 <= q < T:
            raise ServiceError("query position out of range")
        P = A[layer]  # [H, T, T]
        row = P[:, q, :]  # [H, T]
        ent = -(row.clamp_min(1e-12).log() * row).sum(-1)
        z = t["head_z"][layer]  # [T, H, dh]
        out_norm = torch.stack([self.adapter.head_output(layer, h, z[:, h]).norm(dim=-1) for h in range(H)])  # [H, T]
        heads = []
        for h in range(H):
            srcs = torch.topk(row[h, :q + 1], min(5, q + 1))
            heads.append({
                "head": h, "entropy": _r(ent[h], 4), "attn_to_first": _r(row[h, 0], 4),
                "attn_to_self": _r(row[h, q], 4),
                "top_sources": [{"position": int(i), "weight": _r(w, 4)} for i, w in zip(srcs.indices.tolist(), srcs.values.tolist())],
                "output_norm_at_query": _r(out_norm[h, q], 4),
                "kv_group": h // max(1, self.info.n_heads // self.info.n_kv_heads),
            })
        return {
            "run_id": run_id, "layer": layer, "query_position": q, "n_heads": H, "n_tokens": T,
            "patterns": [[[_r(v, 3) for v in r] for r in P[h].tolist()] for h in range(H)],
            "query_rows": _rl(row, 4),
            "head_output_norms": _rl(out_norm, 3),
            "heads": heads,
            "status": {"patterns": OBSERVED, "entropy": DERIVED, "output_norm": DERIVED},
            "caveat": "Attention weights show where information is read from, not why; high attention ≠ causal importance.",
        }

    def head_detail(self, run_id: str, layer: int, head: int, position: int | None = None,
                    target_token_id: int | None = None) -> dict:
        rec = self._get_run(run_id)
        t = self.ensure_level(run_id, "standard")
        tr = self._trace_from_tensors(t, "standard")
        N, H = self.info.n_layers, self.info.n_heads
        if not (0 <= layer < N and 0 <= head < H):
            raise ServiceError("head out of range")
        tgt = rec.get("target") or {}
        pos = tgt.get("position", rec["prompt_len"] - 1) if position is None else int(position)
        tid = target_token_id if target_token_id is not None else tgt.get("token_id")
        proj = head_vocab_projection(self.adapter, tr, layer, head, pos)
        out = self.adapter.head_output(layer, head, tr.head_z[layer, :, head])  # [T, d]
        res: dict[str, Any] = {
            "run_id": run_id, "layer": layer, "head": head, "position": pos,
            "pattern": [[_r(v, 3) for v in r] for r in t["attn"][layer, head].float().tolist()],
            "output_norm_by_position": [_r(v, 4) for v in out.norm(dim=-1).tolist()],
            "promoted": [{"id": i, "text": self._tok_str(i), "value": _r(v, 4)} for i, v in proj["promoted"]],
            "suppressed": [{"id": i, "text": self._tok_str(i), "value": _r(v, 4)} for i, v in proj["suppressed"]],
            "status": {"pattern": OBSERVED, "output_norm": DERIVED, "vocab_projection": DERIVED, "dla": DERIVED},
        }
        if tid is not None:
            u = self.adapter.unembed_direction(int(tid))
            s = float(self.adapter.final_norm_scale(t["resid"][-1, pos]))
            res["dla_to_target"] = {"token": self._tok_str(tid), "value": _r(float(out[pos] @ u) * s, 5)}
        return res

    def dla(self, run_id: str, position: int | None = None, target_token_id: int | None = None,
            contrast_token_id: int | None = None) -> dict:
        rec = self._get_run(run_id)
        t = self.ensure_level(run_id, "standard")
        tr = self._trace_from_tensors(t, "standard")
        tgt = rec.get("target") or {}
        pos = tgt.get("position", rec["prompt_len"] - 1) if position is None else int(position)
        tid = target_token_id if target_token_id is not None else tgt.get("token_id")
        cid = contrast_token_id if contrast_token_id is not None else tgt.get("contrast_id")
        if tid is None:
            raise ServiceError("no target token")
        # actual final logits at pos (needed to verify the decomposition) — recompute from the final residual
        tr.logits = torch.zeros(len(rec["input_ids"]), self.info.vocab_size)
        tr.logits[pos] = self.adapter.logit_lens(t["resid"][-1, pos][None])[0]
        r = direct_logit_attribution(self.adapter, tr, pos, int(tid), None if cid is None else int(cid))
        flat = [(l, h, float(r.heads[l, h])) for l in range(r.heads.shape[0]) for h in range(r.heads.shape[1])]
        flat.sort(key=lambda x: -abs(x[2]))
        return {
            "run_id": run_id, "position": pos, "target": {"id": tid, "text": self._tok_str(tid)},
            "contrast": {"id": cid, "text": self._tok_str(cid)} if cid is not None else None,
            "embed": _r(r.embed), "heads": _rl(r.heads, 4), "mlps": _rl(r.mlps, 4),
            "sum_of_components": _r(r.total), "actual_logit": _r(r.actual),
            "decomposition_error": _r(abs(r.total - r.actual), 6), "final_norm_scale": _r(r.norm_scale, 6),
            "top_heads": [{"layer": l, "head": h, "value": _r(v, 4)} for l, h, v in flat[:12]],
            "status": DERIVED,
            "caveat": "Direct effects only (final-norm scale frozen at its actual value). Components can matter "
                      "indirectly through later layers; use ablation to measure total effects.",
        }

    # ================================================================== interventions
    def run_intervention(self, base_run_id: str, specs: list, max_new_tokens: int | None = None,
                         notes: str | None = None) -> dict:
        base = self._get_run(base_run_id)
        if base["model_id"] != self.info.model_id:
            raise ServiceError(f"run {base_run_id} was produced by {base['model_id']}, loaded model is {self.info.model_id}")
        if not specs:
            raise ServiceError("no interventions given")
        parsed = self._parse_specs(self._dump_specs(specs))
        ids = base["input_ids"]
        P = base["prompt_len"]
        edits = self.compile(parsed, n_seq=len(ids))
        mnt = max_new_tokens or base["generation"]["max_new_tokens"]
        # (1) free-running generation under the intervention → changed output text
        gen = self.adapter.generate(ids[:P], mnt, edits)
        # (2) teacher-forced capture on the BASELINE token sequence under the intervention → aligned trace
        level = base["trace_level"]
        trace = self.adapter.capture(ids, level, edits)
        tensors = self._trace_tensors(trace, P, with_lens=False)
        base_t = self.store.load_tensors(base_run_id)
        comparison = self._intervention_metrics(base, base_t, tensors, trace, gen.generated_ids)
        math_info = None
        correctness = None
        if base.get("math"):
            from neuralscope.mathlab import MathProblem, verify

            prob = MathProblem.from_dict(base["math"]["problem"])
            v = verify(prob, self.adapter.decode(gen.generated_ids))
            math_info = {**base["math"], "verification_under_intervention": v.to_dict()}
            correctness = v.status
            comparison["correctness_before"] = base.get("correctness")
            comparison["correctness_after"] = v.status
        rec = {
            "id": new_id("run"), "created_at": now_iso(), "kind": "intervention", "model_id": base["model_id"],
            "is_demo": base["is_demo"], "prompt": base["prompt"], "prompt_text": base["prompt_text"], "chat": base["chat"],
            "system_prompt": base.get("system_prompt"),
            "generation": {**base["generation"], "max_new_tokens": mnt, "stopped_by_free": gen.stopped_by,
                           "teacher_forced_on": base_run_id},
            "input_ids": ids, "prompt_len": P, "user_span": base.get("user_span"),
            "generated_ids": base["generated_ids"], "generated_text": base["generated_text"],
            "free_generated_ids": gen.generated_ids, "free_generated_text": self.adapter.decode(gen.generated_ids),
            "target": base.get("target"), "trace_level": level, "parent_run_id": base_run_id,
            "interventions": [s.model_dump() for s in parsed], "metrics": {"comparison": comparison},
            "math": math_info, "correctness": correctness, "notes": notes, "tags": [],
            "benchmark_id": None, "trace_consistent": None,
        }
        saved = self.store.save_run(rec, tensors)
        return {"run": self.run_summary(saved), "comparison": comparison}

    def _intervention_metrics(self, base: dict, base_t: dict, int_t: dict, int_trace: Trace,
                              free_ids: list[int]) -> dict:
        P = base["prompt_len"]
        ids = base["input_ids"]
        tgt = base.get("target") or {}
        pos = tgt.get("position", P - 1)
        lp_b_all = base_t["gen_logprobs"]  # positions P-1 .. T-1
        lp_i_all = int_t["gen_logprobs"]
        per_pos = []
        for j in range(lp_b_all.shape[0]):
            p = P - 1 + j
            nb, ni = lp_b_all[j], lp_i_all[j]
            nxt = ids[p + 1] if p + 1 < len(ids) else None
            per_pos.append({
                "position": p,
                "next_token": self._tok_str(nxt) if nxt is not None else None,
                "kl": _r(kl_divergence(nb, ni), 6),
                "base_logprob": _r(nb[nxt]) if nxt is not None else None,
                "int_logprob": _r(ni[nxt]) if nxt is not None else None,
                "base_top1": self._tok_str(int(nb.argmax())), "int_top1": self._tok_str(int(ni.argmax())),
                "top1_changed": int(nb.argmax()) != int(ni.argmax()),
            })
        # target metrics at the target position (logits recomputed from final residual for exactness)
        lg_b = self.adapter.logit_lens(base_t["resid"][-1, pos][None])[0]
        lg_i = int_trace.logits[pos]
        lpb, lpi = torch.log_softmax(lg_b, -1), torch.log_softmax(lg_i, -1)
        tid = tgt.get("token_id")
        cid = tgt.get("contrast_id")

        def stats(lg, lp):
            if tid is None:
                return None
            s = {"logit": _r(lg[tid]), "prob": _r(lp[tid].exp(), 6), "logprob": _r(lp[tid]),
                 "rank": int(1 + (lg > lg[tid]).sum())}
            if cid is not None:
                s["logit_diff"] = _r(lg[tid] - lg[cid])
            return s

        before, after = stats(lg_b, lpb), stats(lg_i, lpi)
        delta = None
        if before and after:
            delta = {k: _r(after[k] - before[k], 6) for k in ("logit", "prob", "logprob") if before[k] is not None}
            delta["rank"] = after["rank"] - before["rank"]
            if "logit_diff" in before:
                delta["logit_diff"] = _r(after["logit_diff"] - before["logit_diff"], 6)
        topk = lambda lg, lp: [{"id": int(i), "text": self._tok_str(i), "prob": _r(lp[i].exp(), 6), "logit": _r(lg[i])}
                               for i in torch.topk(lg, 10).indices.tolist()]
        base_gen = base["generated_ids"]
        changed = [i for i in range(max(len(base_gen), len(free_ids)))
                   if i >= len(base_gen) or i >= len(free_ids) or base_gen[i] != free_ids[i]]
        rd = relative_delta(base_t["resid"], int_t["resid"])  # [D, T]
        by_depth = rd.max(dim=1).values
        affected = [d for d in range(rd.shape[0]) if float(by_depth[d]) > 1e-6]
        return {
            "target": tgt, "target_position": pos,
            "before": before, "after": after, "delta": delta,
            "kl_at_target": _r(kl_divergence(lpb, lpi), 6),
            "max_kl": _r(max((x["kl"] or 0.0) for x in per_pos), 6) if per_pos else None,
            "top_before": topk(lg_b, lpb), "top_after": topk(lg_i, lpi),
            "per_position": per_pos,
            "output_before": base["generated_text"], "output_after": self.adapter.decode(free_ids),
            "output_changed": base_gen != free_ids, "changed_token_indices": changed,
            "first_changed_step": changed[0] if changed else None,
            "resid_delta_by_depth": [_r(v, 6) for v in by_depth.tolist()],
            "earliest_affected_depth": affected[0] if affected else None,
            "status": CAUSAL,
            "scope": "single prompt (n=1); teacher-forced on baseline tokens for aligned metrics, free-running for output",
        }

    # ================================================================== cross-run comparison
    def compare(self, run_a: str, run_b: str, position_a: int | None = None, position_b: int | None = None) -> dict:
        A, B = self._get_run(run_a), self._get_run(run_b)
        if B.get("parent_run_id") == run_a and (B.get("metrics") or {}).get("comparison"):
            return {"mode": "intervention", "run_a": self.run_summary(A), "run_b": self.run_summary(B),
                    "comparison": B["metrics"]["comparison"]}
        ta, tb = self.store.load_tensors(run_a), self.store.load_tensors(run_b)
        ga, gb = A.get("target") or {}, B.get("target") or {}
        pa = ga.get("position", A["prompt_len"] - 1) if position_a is None else int(position_a)
        pb = gb.get("position", B["prompt_len"] - 1) if position_b is None else int(position_b)
        Ra, Rb = ta["resid"], tb["resid"]
        D = Ra.shape[0]
        ha, hb = Ra[:, pa], Rb[:, pb]
        la, lb = self.adapter.logit_lens(ha), self.adapter.logit_lens(hb)  # [D, V]
        lpa, lpb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)

        def series(lg, lp, R, p, tid):
            s = {
                "resid_norm": _rl(R[:, p].norm(dim=-1), 4),
                "cos_change": [None] + _rl(1 - cosine(R[1:, p], R[:-1, p]), 5),
                "lens_entropy": _rl(entropy(lp), 4),
                "lens_top1": [self._tok_str(i) for i in lg.argmax(-1).tolist()],
                "lens_top1_prob": _rl(lp.max(-1).values.exp(), 4),
            }
            if tid is not None:
                s["target_rank"] = [int(x) for x in (1 + (lg > lg[:, tid:tid + 1]).sum(-1)).tolist()]
                s["target_prob"] = _rl(lp[:, tid].exp(), 5)
            return s

        out: dict[str, Any] = {
            "mode": "cross_run",
            "run_a": self.run_summary(A), "run_b": self.run_summary(B),
            "position_a": pa, "position_b": pb,
            "token_a": self._tok_str(A["input_ids"][pa]), "token_b": self._tok_str(B["input_ids"][pb]),
            "target_a": ga, "target_b": gb,
            "depth_labels": self.depth_labels(),
            "series_a": series(la, lpa, Ra, pa, ga.get("token_id")),
            "series_b": series(lb, lpb, Rb, pb, gb.get("token_id")),
            "cross_cosine": _rl(cosine(ha, hb), 5),
            "final_kl_a_b": _r(kl_divergence(lpa[-1], lpb[-1]), 5),
            "status": DERIVED,
            "caveat": "Correlational: differences between two runs on different inputs are not evidence that any "
                      "difference *causes* the behavioural difference. Test candidates with patching or ablation.",
        }
        # attention differences at the compared positions (lazy capture if needed)
        try:
            ta2, tb2 = self.ensure_level(run_a, "standard"), self.ensure_level(run_b, "standard")
            Aa, Ab = ta2["attn"].float()[:, :, pa, :pa + 1], tb2["attn"].float()[:, :, pb, :pb + 1]
            ea = -(Aa.clamp_min(1e-12).log() * Aa).sum(-1)
            eb = -(Ab.clamp_min(1e-12).log() * Ab).sum(-1)
            out["attn_entropy_a"] = _rl(ea, 4)
            out["attn_entropy_b"] = _rl(eb, 4)
            out["attn_entropy_diff"] = _rl(ea - eb, 4)
        except Exception as e:  # pragma: no cover — attention optional
            out["attn_error"] = str(e)
        same_len = len(A["input_ids"]) == len(B["input_ids"])
        out["same_length"] = same_len
        if same_len:
            out["token_diff_positions"] = [i for i, (x, y) in enumerate(zip(A["input_ids"], B["input_ids"])) if x != y]
            out["resid_rel_diff"] = _rl(relative_delta(Ra, Rb), 4)
        return out

    # ================================================================== sweeps (CAUSAL, batched)
    def head_ablation_sweep(self, run_id: str, position: int | None = None, target_token_id: int | None = None,
                            contrast_token_id: int | None = None, factor: float = 0.0,
                            progress: Progress | None = None) -> dict:
        """Zero-ablate every head (one at a time, all positions) and measure the effect at ``position``."""
        rec = self._get_run(run_id)
        tgt = rec.get("target") or {}
        pos = tgt.get("position", rec["prompt_len"] - 1) if position is None else int(position)
        tid = target_token_id if target_token_id is not None else tgt.get("token_id")
        cid = contrast_token_id if contrast_token_id is not None else tgt.get("contrast_id")
        if tid is None:
            raise ServiceError("sweep needs a target token")
        ids = rec["input_ids"][:pos + 1]  # causal: later tokens cannot affect position `pos`
        edits0 = self.compile(rec.get("interventions") or [], n_seq=len(ids))
        N, H = self.info.n_layers, self.info.n_heads
        base_lg = self.adapter.final_logits_batch(ids, [edits0])[0, pos]
        base_lp = torch.log_softmax(base_lg, -1)

        def metric(lg):
            return float(lg[tid] - lg[cid]) if cid is not None else float(torch.log_softmax(lg, -1)[tid])

        m0 = metric(base_lg)
        d_metric = torch.zeros(N, H)
        d_prob = torch.zeros(N, H)
        kl = torch.zeros(N, H)
        t0 = time.time()
        R = self.store.load_tensors(run_id)["resid"]
        for l in range(N):
            # resume from the stored residual entering block l: blocks < l are unaffected by the ablation
            pre = self._edits_from_depth(edits0, l)
            sets = [pre + [HeadScaleEdit(l, h, factor, None)] for h in range(H)]
            lg = self.adapter.final_logits_batch(ids, sets, start_depth=l, resid_in=R[l, :pos + 1])[:, pos]  # [H, V]
            lp = torch.log_softmax(lg, -1)
            for h in range(H):
                d_metric[l, h] = metric(lg[h]) - m0
                d_prob[l, h] = float(lp[h, tid].exp() - base_lp[tid].exp())
                kl[l, h] = float(kl_divergence(base_lp, lp[h]))
            if progress:
                progress((l + 1) / N, f"layer {l + 1}/{N}")
        flat = sorted(((l, h, float(d_metric[l, h])) for l in range(N) for h in range(H)), key=lambda x: x[2])
        result = {
            "id": new_id("sweep"), "kind": "head_ablation", "run_id": run_id, "position": pos,
            "target": {"id": tid, "text": self._tok_str(tid)},
            "contrast": {"id": cid, "text": self._tok_str(cid)} if cid is not None else None,
            "metric": "logit_diff" if cid is not None else "target_logprob", "baseline_metric": _r(m0),
            "baseline_prob": _r(base_lp[tid].exp(), 6), "factor": factor,
            "delta_metric": _rl(d_metric, 5), "delta_prob": _rl(d_prob, 6), "kl": _rl(kl, 6),
            "most_harmful": [{"layer": l, "head": h, "delta": _r(v, 5)} for l, h, v in flat[:10]],
            "most_helpful": [{"layer": l, "head": h, "delta": _r(v, 5)} for l, h, v in flat[::-1][:10]],
            "seconds": round(time.time() - t0, 2), "created_at": now_iso(),
            "status": CAUSAL, "scope": "single prompt (n=1), one head at a time, all positions",
        }
        m = dict(rec.get("metrics") or {})
        m.setdefault("sweeps", []).append(result)
        self.store.update_run(run_id, metrics=m)
        return result

    def patch_sweep(self, clean_run_id: str, corrupt_run_id: str, progress: Progress | None = None,
                    positions: list[int] | None = None) -> dict:
        """Residual activation patching: for each (depth, position), h_corrupt ← h_clean and measure restoration.

        Metric at the corrupt run's target position: LD = logit(τ_clean) − logit(τ_corrupt).
        Restoration = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt).
        Requires equal-length token sequences (position-aligned counterfactual pair).
        """
        C, X = self._get_run(clean_run_id), self._get_run(corrupt_run_id)
        if len(C["input_ids"]) != len(X["input_ids"]) or C["prompt_len"] != X["prompt_len"]:
            raise ServiceError("activation patching sweep needs two runs with identical token lengths and prompt lengths "
                               "(e.g. '37 + 58' vs '37 + 59'); use a single ResidualPatch for other pairs")
        pos = (X.get("target") or {}).get("position", X["prompt_len"] - 1)
        tc = (C.get("target") or {}).get("token_id")
        tx = (X.get("target") or {}).get("token_id")
        if tc is None or tx is None:
            raise ServiceError("both runs need a target token")
        ids = X["input_ids"][:pos + 1]
        Rc = self.store.load_tensors(clean_run_id)["resid"]
        Rx = self.store.load_tensors(corrupt_run_id)["resid"]
        if X.get("interventions") or C.get("interventions"):
            raise ServiceError("patch sweeps run between un-intervened runs")
        diff_positions = [i for i, (a, b) in enumerate(zip(C["input_ids"][:pos + 1], ids)) if a != b]
        if positions is None:
            first = diff_positions[0] if diff_positions else pos
            positions = list(range(first, pos + 1))
        D = self.info.n_depths

        def ld(lg):
            return float(lg[tc] - lg[tx]) if tc != tx else float(torch.log_softmax(lg, -1)[tc])

        lg_x = self.adapter.final_logits_batch(ids, [[]])[0, pos]
        lg_c = self.adapter.final_logits_batch(C["input_ids"][:pos + 1], [[]])[0, pos]
        m_x, m_c = ld(lg_x), ld(lg_c)
        denom = m_c - m_x
        rest = torch.full((D, len(ids)), float("nan"))
        raw = torch.full((D, len(ids)), float("nan"))
        t0 = time.time()
        for d in range(D):
            sets = [[ResidSetEdit(d, p, Rc[d, p].clone(), 1.0)] for p in positions]
            lg = self.adapter.final_logits_batch(ids, sets, start_depth=d, resid_in=Rx[d, :pos + 1])[:, pos]
            for j, p in enumerate(positions):
                v = ld(lg[j])
                raw[d, p] = v
                rest[d, p] = (v - m_x) / denom if abs(denom) > 1e-6 else float("nan")
            if progress:
                progress((d + 1) / D, f"depth {d + 1}/{D}")
        result = {
            "id": new_id("sweep"), "kind": "resid_patch", "clean_run_id": clean_run_id, "corrupt_run_id": corrupt_run_id,
            "position": pos, "positions": positions, "diff_positions": diff_positions,
            "clean_target": self._tok_str(tc), "corrupt_target": self._tok_str(tx),
            "metric": "logit(τ_clean) − logit(τ_corrupt)" if tc != tx else "logprob(τ)",
            "metric_clean": _r(m_c), "metric_corrupt": _r(m_x),
            "restoration": _rl(rest, 4), "raw_metric": _rl(raw, 4),
            "tokens": [self._tok_str(i) for i in ids], "depth_labels": self.depth_labels(),
            "seconds": round(time.time() - t0, 2), "created_at": now_iso(),
            "status": CAUSAL, "scope": "single counterfactual pair (n=1)",
        }
        m = dict(X.get("metrics") or {})
        m.setdefault("sweeps", []).append(result)
        self.store.update_run(corrupt_run_id, metrics=m)
        return result

    # ================================================================== Math Lab
    def answer_logprob(self, prompt_ids: list[int], answer_text: str, edits: list[Edit] | None = None) -> dict:
        """Teacher-forced log P(answer tokens | prompt) (+ end-of-turn not included)."""
        ans = self.adapter.encode(answer_text)
        ids = prompt_ids + ans
        lg = self.adapter.final_logits_batch(ids, [edits or []])[0]
        lp = torch.log_softmax(lg, -1)
        P = len(prompt_ids)
        per = [float(lp[P - 1 + i, a]) for i, a in enumerate(ans)]
        return {"tokens": self.adapter.token_strings(ans), "token_ids": ans, "per_token": [_r(x) for x in per],
                "total": _r(sum(per))}

    def run_math_problem(self, problem, benchmark_id: str | None = None, trace_level: TraceLevel = "minimal",
                         max_new_tokens: int = 24) -> dict:
        from neuralscope.mathlab import verify
        from neuralscope.mathlab.alignment import first_divergence

        tp = self.adapter.tokenize(problem.question, chat=True)
        gen = self.adapter.generate(tp.input_ids, max_new_tokens)
        text = self.adapter.decode(gen.generated_ids)
        v = verify(problem, text)
        # An output cut off by the token budget may look like a finished (wrong) answer ("10" from "1022").
        # Such runs are labelled 'truncated' and excluded from correct-vs-incorrect comparisons.
        truncated = gen.stopped_by != "eos"
        label = "truncated" if truncated else v.status
        oracle_lp = self.answer_logprob(tp.input_ids, problem.oracle_answer)
        # locate the decision point: first generated token where the model departs from the oracle answer
        P = len(tp.input_ids)
        oracle_ids = self.adapter.encode(problem.oracle_answer)
        target_pos, target_id, contrast_id, how = P - 1, oracle_ids[0], None, "first answer token"
        k = 0
        while k < min(len(oracle_ids), len(gen.generated_ids)) and oracle_ids[k] == gen.generated_ids[k]:
            k += 1
        if label == "incorrect" and k < len(oracle_ids) and k < len(gen.generated_ids):
            target_pos, target_id, contrast_id = P - 1 + k, oracle_ids[k], gen.generated_ids[k]
            how = f"first divergent answer token (index {k}): oracle '{self._tok_str(oracle_ids[k])}' vs model '{self._tok_str(gen.generated_ids[k])}'"
        math_info = {
            "problem": problem.to_dict(), "verification": v.to_dict(),
            "divergence": first_divergence(v.parsed_answer, problem.oracle_answer),
            "oracle_answer_logprob": oracle_lp,
            "decision_point": {"position": target_pos, "k": k, "how": how},
            "truncated": truncated, "label": label,
        }
        rec = self._create_from_generation(tp, gen, trace_level, kind="math", math=math_info,
                                           benchmark_id=benchmark_id, correctness=label,
                                           target=dict(position=target_pos, token_id=target_id,
                                                       contrast_id=contrast_id))
        return rec

    def _create_from_generation(self, tp, gen, level, kind, math, benchmark_id, correctness, target) -> dict:
        ids = tp.input_ids + gen.generated_ids
        trace = self.adapter.capture(ids, level)
        P = len(tp.input_ids)
        consistent = all(int(trace.logits[P - 1 + i].argmax()) == g for i, g in enumerate(gen.generated_ids))
        tensors = self._trace_tensors(trace, P, with_lens=False)
        tgt = self.resolve_target(ids, P, target_token_id=target["token_id"], target_position=target["position"],
                                  contrast_token_id=target.get("contrast_id"))
        tgt["source"] = "oracle"
        rec = {
            "id": new_id("run"), "created_at": now_iso(), "kind": kind, "model_id": self.info.model_id,
            "is_demo": self.info.is_demo, "prompt": math["problem"]["question"], "prompt_text": tp.text, "chat": True,
            "system_prompt": None,
            "generation": {"decoding": "greedy", "max_new_tokens": len(gen.generated_ids), "stopped_by": gen.stopped_by,
                           "kv_cache": True, "attn_implementation": "eager", "dtype": self.info.dtype},
            "input_ids": ids, "prompt_len": P, "user_span": list(tp.user_span) if tp.user_span else None,
            "generated_ids": gen.generated_ids, "generated_text": self.adapter.decode(gen.generated_ids),
            "target": tgt, "trace_level": level, "parent_run_id": None, "interventions": [], "metrics": {},
            "math": math, "correctness": correctness, "notes": None, "tags": ["mathlab"],
            "benchmark_id": benchmark_id, "trace_consistent": consistent,
        }
        return self.store.save_run(rec, tensors)

    def run_benchmark(self, problems: list, name: str = "Math Lab benchmark", seed: int | None = None,
                      progress: Progress | None = None) -> dict:
        bench = self.store.save_benchmark({
            "id": new_id("bench"), "created_at": now_iso(), "model_id": self.info.model_id,
            "is_demo": int(self.info.is_demo), "name": name,
            "spec": {"n": len(problems), "seed": seed, "problems": [p.to_dict() for p in problems]},
            "results": [], "summary": None, "status": "running"})
        results = []
        t0 = time.time()
        for i, prob in enumerate(problems):
            rec = self.run_math_problem(prob, benchmark_id=bench["id"])
            m = rec["math"]
            results.append({
                "problem_id": prob.id, "kind": prob.kind, "difficulty": prob.difficulty, "question": prob.question,
                "oracle_answer": prob.oracle_answer, "model_output": rec["generated_text"],
                "parsed_answer": m["verification"]["parsed_answer"], "status": m["label"],
                "correct": m["label"] == "correct", "run_id": rec["id"],
                "oracle_logprob": m["oracle_answer_logprob"]["total"],
            })
            if progress:
                progress((i + 1) / len(problems), f"{i + 1}/{len(problems)} · {prob.question}")
        summary = self._bench_summary(results)
        summary["seconds"] = round(time.time() - t0, 1)
        bench["results"], bench["summary"], bench["status"] = results, summary, "done"
        bench["is_demo"] = int(bench["is_demo"])
        return self.store.save_benchmark(bench)

    @staticmethod
    def _bench_summary(results: list[dict]) -> dict:
        n = len(results)
        c = sum(1 for r in results if r["correct"])
        by_kind: dict[str, dict] = {}
        for r in results:
            k = by_kind.setdefault(r["kind"], {"n": 0, "correct": 0})
            k["n"] += 1
            k["correct"] += int(r["correct"])
        for k in by_kind.values():
            k["accuracy"] = round(k["correct"] / k["n"], 4)
        return {"n": n, "correct": c, "incorrect": sum(1 for r in results if r["status"] == "incorrect"),
                "unparseable": sum(1 for r in results if r["status"] in ("unparseable", "ambiguous")),
                "truncated": sum(1 for r in results if r["status"] == "truncated"),
                "accuracy": round(c / n, 4) if n else None, "by_kind": by_kind, "status": OBSERVED}

    def benchmark_intervention_eval(self, benchmark_id: str, specs: list, progress: Progress | None = None) -> dict:
        """Apply the same intervention to every benchmark problem: accuracy + oracle-answer log-prob shift (CAUSAL, n=N)."""
        from neuralscope.mathlab import MathProblem, verify

        bench = self.store.get_benchmark(benchmark_id)
        if not bench:
            raise KeyError(benchmark_id)
        parsed = self._parse_specs(self._dump_specs(specs))
        if any(isinstance(s, ResidualPatch) for s in parsed):
            raise ServiceError("position-specific residual patches are not transferable across problems; "
                               "use head/MLP ablation or steering")
        rows = []
        t0 = time.time()
        for i, r in enumerate(bench["results"]):
            run = self._get_run(r["run_id"])
            prob = MathProblem.from_dict(run["math"]["problem"])
            P = run["prompt_len"]
            edits = self.compile(parsed, n_seq=None)
            gen = self.adapter.generate(run["input_ids"][:P], max(8, len(run["generated_ids"]) + 2), edits)
            v = verify(prob, self.adapter.decode(gen.generated_ids))
            lp = self.answer_logprob(run["input_ids"][:P], prob.oracle_answer, edits)
            rows.append({"problem_id": prob.id, "question": prob.question, "run_id": run["id"],
                         "correct_before": r["correct"], "correct_after": v.correct, "status_after": v.status,
                         "output_after": self.adapter.decode(gen.generated_ids),
                         "oracle_logprob_before": r["oracle_logprob"], "oracle_logprob_after": lp["total"],
                         "delta_oracle_logprob": _r((lp["total"] or 0) - (r["oracle_logprob"] or 0))})
            if progress:
                progress((i + 1) / len(bench["results"]), f"{i + 1}/{len(bench['results'])}")
        n = len(rows)
        deltas = torch.tensor([x["delta_oracle_logprob"] for x in rows], dtype=torch.float64)
        acc_b = sum(x["correct_before"] for x in rows) / n
        acc_a = sum(x["correct_after"] for x in rows) / n
        res = {
            "id": new_id("eval"), "benchmark_id": benchmark_id, "interventions": [s.model_dump() for s in parsed],
            "interventions_label": describe_all(parsed), "n": n,
            "accuracy_before": round(acc_b, 4), "accuracy_after": round(acc_a, 4),
            "flips_correct_to_incorrect": sum(1 for x in rows if x["correct_before"] and not x["correct_after"]),
            "flips_incorrect_to_correct": sum(1 for x in rows if not x["correct_before"] and x["correct_after"]),
            "mean_delta_oracle_logprob": _r(deltas.mean()), "sd_delta_oracle_logprob": _r(deltas.std()) if n > 1 else None,
            "sem_delta_oracle_logprob": _r(deltas.std() / math.sqrt(n)) if n > 1 else None,
            "rows": rows, "seconds": round(time.time() - t0, 1), "created_at": now_iso(),
            "status": CAUSAL, "scope": f"{n} problems from benchmark {benchmark_id}",
        }
        summary = dict(bench.get("summary") or {})
        summary.setdefault("intervention_evals", []).append({k: v for k, v in res.items() if k != "rows"})
        bench["summary"] = summary
        evals = dict(bench.get("spec") or {})
        bench["spec"] = evals
        self.store.save_benchmark({**bench, "is_demo": int(bench["is_demo"])})
        return res
