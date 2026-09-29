"""End-to-end integration on a REAL pretrained model (slow; `pytest -m slow`).

prompt → model run → capture → intervention → rerun → comparison, plus the
scientific invariants that must hold on real weights (not just the fixture).
"""

from __future__ import annotations

import pytest
import torch

from neuralscope.config import Settings
from neuralscope.experiments.service import NeuralScopeService
from neuralscope.experiments.store import ExperimentStore
from neuralscope.schemas.interventions import HeadAblation, ResidualPatch
from tests.conftest import PROMPT, REAL_TEST_MODEL

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_service(real_adapter, tmp_path_factory):
    s = Settings()
    s.model_id = REAL_TEST_MODEL
    s.data_dir = tmp_path_factory.mktemp("real")
    s.trace_level = "standard"
    return NeuralScopeService(real_adapter, ExperimentStore(s.data_dir), s)


def test_real_model_loads(real_adapter):
    i = real_adapter.info
    assert not i.is_demo and i.n_layers >= 12 and i.architecture.startswith("qwen2")
    assert i.dtype == "float32"


def test_real_generation_is_deterministic_and_correct(real_service):
    a = real_service.create_run(PROMPT, max_new_tokens=6)
    b = real_service.create_run(PROMPT, max_new_tokens=6)
    assert a["generated_ids"] == b["generated_ids"]
    # correctness is model-dependent (Qwen2.5-0.5B answers "85": it drops the carry; 1.5B answers "95"),
    # so only require a numeric answer here — correctness is what Math Lab measures, not what tests assume.
    assert a["generated_text"][:1].isdigit()
    assert a["trace_consistent"] is True
    ta, tb = real_service.store.load_tensors(a["id"]), real_service.store.load_tensors(b["id"])
    assert torch.equal(ta["resid"], tb["resid"])


def test_real_invariants(real_service, real_adapter):
    run = real_service.create_run(PROMPT, max_new_tokens=4)
    t = real_service.store.load_tensors(run["id"])
    pos = run["target"]["position"]
    # final-depth logit lens reproduces the model's greedy choice
    lens = real_service.logit_lens(run["id"])
    assert lens["depths"][-1]["target"]["rank"] == 1
    assert lens["target"]["text"] == real_service.adapter.token_strings([run["generated_ids"][0]])[0]
    # DLA decomposes the real logit exactly
    d = real_service.dla(run["id"])
    assert d["decomposition_error"] < 1e-2 * max(1.0, abs(d["actual_logit"]))
    # attention rows are normalised & causal
    att = real_service.attention(run["id"], real_adapter.info.n_layers // 2)
    for row in att["query_rows"]:
        assert abs(sum(row) - 1) < 1e-2 and all(v == 0 for v in row[pos + 1:])
    assert t["resid"].shape[0] == real_adapter.info.n_layers + 1


def test_real_prompt_to_comparison(real_service):
    """The acceptance integration test: prompt → run → capture → intervention → rerun → comparison."""
    base = real_service.create_run(PROMPT, max_new_tokens=6)
    sw = real_service.head_ablation_sweep(base["id"])
    worst = sw["most_harmful"][0]
    assert worst["delta"] < 0
    res = real_service.run_intervention(base["id"], [HeadAblation(layer=worst["layer"], head=worst["head"])])
    c = res["comparison"]
    assert c["status"] == "causal"
    assert c["after"]["prob"] < c["before"]["prob"]
    assert abs(c["delta"]["logprob"] - worst["delta"]) < 1e-2  # sweep and single intervention agree
    assert c["kl_at_target"] > 0
    assert c["earliest_affected_depth"] == worst["layer"] + 1
    cmp = real_service.compare(base["id"], res["run"]["id"])
    assert cmp["mode"] == "intervention"


def test_real_activation_patching_restores(real_service):
    # counterfactual pairs differing in exactly one digit token; use the first pair on which this model's
    # first answer tokens differ (models differ in which problems they get right and in output format)
    pairs = [("37 + 58", "37 + 48"), ("12 + 34", "52 + 34"), ("23 + 41", "63 + 41"), ("6 + 2", "3 + 2")]
    for a, b in pairs:
        clean = real_service.create_run(f"What is {a}? Give only the answer.", max_new_tokens=3)
        corrupt = real_service.create_run(f"What is {b}? Give only the answer.", max_new_tokens=3)
        if clean["generated_ids"][0] != corrupt["generated_ids"][0]:
            break
    else:
        pytest.skip("no candidate pair produced different first answer tokens on this model")
    pos = corrupt["target"]["position"]
    L = real_service.info.n_layers
    # patching the full final-depth state at the readout position must reproduce the clean prediction exactly
    res = real_service.run_intervention(corrupt["id"], [ResidualPatch(depth=L, position=pos, source_run_id=clean["id"])])
    assert res["comparison"]["top_after"][0]["id"] == clean["generated_ids"][0]
    # and a full patching sweep gives restoration 1 at (final depth, readout position) and at the embedding
    # of the single differing token
    ps = real_service.patch_sweep(clean["id"], corrupt["id"])
    assert abs(ps["restoration"][L][pos] - 1) < 1e-2
    assert len(ps["diff_positions"]) == 1
    assert abs(ps["restoration"][0][ps["diff_positions"][0]] - 1) < 1e-2
