"""Service layer: runs, views, interventions, comparisons, sweeps, persistence (fixture model)."""

from __future__ import annotations

import pytest
import torch

from neuralscope.experiments.service import NeuralScopeService, ServiceError
from neuralscope.experiments.store import ExperimentStore
from neuralscope.schemas.interventions import HeadAblation, MLPAblation, ResidualPatch, ResidualSteer
from tests.conftest import PROMPT


@pytest.fixture()
def run(service):
    return service.create_run(PROMPT, max_new_tokens=4)


def test_create_run_record(service, run):
    assert run["is_demo"] is True and run["kind"] == "baseline"
    assert run["trace_consistent"] is True
    assert len(run["input_ids"]) == run["prompt_len"] + len(run["generated_ids"])
    assert run["target"]["position"] == run["prompt_len"] - 1
    assert run["target"]["token_id"] == run["input_ids"][run["prompt_len"]]
    meta = run["tensor_meta"]["keys"]
    D, T = service.info.n_layers + 1, len(run["input_ids"])
    assert meta["resid"]["shape"] == [D, T, service.info.d_model]
    assert meta["attn"]["shape"] == [service.info.n_layers, service.info.n_heads, T, T]
    assert meta["gen_logprobs"]["shape"][0] == T - run["prompt_len"] + 1


def test_run_detail_tokens(service, run):
    d = service.run_detail(run["id"])
    roles = {t["role"] for t in d["tokens"]}
    assert {"template", "user", "generated"} <= roles
    user = "".join(t["text"] for t in d["tokens"] if t["role"] == "user")
    assert user == PROMPT
    assert d["depth_labels"][0] == "emb" and len(d["depth_labels"]) == service.info.n_layers + 1
    assert all(len(t["top"]) == 5 for t in d["tokens"])


@pytest.mark.parametrize("metric", ["lens_top1_prob", "lens_entropy", "target_prob", "target_rank",
                                    "next_token_rank", "resid_norm", "cos_delta"])
def test_matrix_metrics(service, run, metric):
    m = service.matrix(run["id"], metric)
    assert len(m["values"]) == service.info.n_layers + 1
    assert len(m["values"][0]) == len(run["input_ids"])
    assert m["status"] == "derived"


def test_matrix_target_rank_final_depth(service, run):
    m = service.matrix(run["id"], "target_rank")
    pos = run["target"]["position"]
    assert m["ranks"][-1][pos] == 1  # greedy output is the final-depth argmax


def test_delta_resid_requires_intervention(service, run):
    with pytest.raises(ServiceError):
        service.matrix(run["id"], "delta_resid")


def test_cell_and_lens(service, run):
    pos = run["target"]["position"]
    c = service.cell(run["id"], service.info.n_layers, pos)
    assert c["vector"]["status"] == "observed" and len(c["vector"]["values"]) == service.info.d_model
    assert c["lens"]["top"][0]["id"] == run["target"]["token_id"]
    assert c["target"]["lens_rank"] == 1
    L = service.logit_lens(run["id"])
    assert len(L["depths"]) == service.info.n_layers + 1
    assert L["depths"][-1]["target"]["rank"] == 1
    assert L["first_depth_target_top1"] is not None
    with pytest.raises(ServiceError):
        service.cell(run["id"], 99, 0)


def test_attention_and_head_views(service, run):
    A = service.attention(run["id"], 1)
    H, T = service.info.n_heads, len(run["input_ids"])
    assert len(A["patterns"]) == H and len(A["patterns"][0]) == T
    q = A["query_position"]
    for h in range(H):
        assert abs(sum(A["query_rows"][h]) - 1) < 1e-2
        assert all(v == 0 for v in A["query_rows"][h][q + 1:])
    hd = service.head_detail(run["id"], 1, 2)
    assert len(hd["pattern"]) == T and "dla_to_target" in hd and len(hd["promoted"]) == 8


def test_dla_view(service, run):
    d = service.dla(run["id"])
    assert d["decomposition_error"] < 1e-3
    assert len(d["heads"]) == service.info.n_layers and len(d["heads"][0]) == service.info.n_heads


def test_head_ablation_intervention(service, run):
    res = service.run_intervention(run["id"], [HeadAblation(layer=2, head=1)])
    c = res["comparison"]
    assert c["status"] == "causal"
    assert c["kl_at_target"] >= 0
    assert abs(c["delta"]["logit"] - (c["after"]["logit"] - c["before"]["logit"])) < 1e-4
    assert c["earliest_affected_depth"] == 3
    assert all(v == 0 for v in c["resid_delta_by_depth"][:3])
    child = service.store.get_run(res["run"]["id"])
    assert child["parent_run_id"] == run["id"] and child["input_ids"] == run["input_ids"]
    m = service.matrix(child["id"], "delta_resid")
    assert m["status"] == "causal" and max(m["values"][1]) == 0
    # compare endpoint returns the stored intervention comparison
    cmp = service.compare(run["id"], child["id"])
    assert cmp["mode"] == "intervention"
    # the parent lists the child
    assert service.run_detail(run["id"])["children"][0]["id"] == child["id"]


def test_intervention_validation(service, run):
    with pytest.raises(ServiceError):
        service.run_intervention(run["id"], [HeadAblation(layer=99, head=0)])
    with pytest.raises(ServiceError):
        service.run_intervention(run["id"], [])


def test_patch_and_steer_interventions(service, run):
    other = service.create_run("What is 37 + 59? Give only the answer.", max_new_tokens=4)
    pos = run["target"]["position"]
    res = service.run_intervention(other["id"], [ResidualPatch(depth=service.info.n_layers, position=pos,
                                                               source_run_id=run["id"])])
    # patching the final-depth residual at the target position reproduces the clean run's prediction exactly
    c = res["comparison"]
    clean_top = run["target"]["token_id"]
    assert c["top_after"][0]["id"] == clean_top
    st = service.run_intervention(other["id"], [ResidualSteer(depth=2, plus_run_id=run["id"], plus_position=pos,
                                                               minus_run_id=other["id"], minus_position=pos,
                                                               positions=[pos])])
    assert st["comparison"]["earliest_affected_depth"] == 2
    ml = service.run_intervention(run["id"], [MLPAblation(layer=0)])
    assert ml["comparison"]["earliest_affected_depth"] == 1


def test_cross_run_compare(service, run):
    other = service.create_run("What is 12 + 34? Give only the answer.", max_new_tokens=4)
    cmp = service.compare(run["id"], other["id"])
    assert cmp["mode"] == "cross_run" and cmp["status"] == "derived"
    D = service.info.n_layers + 1
    assert len(cmp["cross_cosine"]) == D and len(cmp["series_a"]["resid_norm"]) == D
    assert "attn_entropy_diff" in cmp
    if cmp["same_length"]:
        assert cmp["token_diff_positions"]


def test_head_sweep_matches_individual_intervention(service, run):
    sw = service.head_ablation_sweep(run["id"])
    N, H = service.info.n_layers, service.info.n_heads
    assert len(sw["delta_metric"]) == N and len(sw["delta_metric"][0]) == H
    l, h = sw["most_harmful"][0]["layer"], sw["most_harmful"][0]["head"]
    iv = service.run_intervention(run["id"], [HeadAblation(layer=l, head=h)])
    assert abs(iv["comparison"]["delta"]["logprob"] - sw["delta_metric"][l][h]) < 1e-3
    assert service.store.get_run(run["id"])["metrics"]["sweeps"][0]["kind"] == "head_ablation"


def test_patch_sweep_restoration_bounds(service):
    clean = service.create_run("What is 37 + 58? Give only the answer.", max_new_tokens=3)
    corrupt = service.create_run("What is 37 + 48? Give only the answer.", max_new_tokens=3)
    if clean["target"]["token_id"] == corrupt["target"]["token_id"]:
        # random-weight model may predict the same token; force distinct targets for the test
        service.store.update_run(clean["id"], target={**clean["target"], "token_id": clean["target"]["token_id"]})
    ps = service.patch_sweep(clean["id"], corrupt["id"])
    pos = ps["position"]
    D = service.info.n_layers + 1
    diff = ps["diff_positions"]
    assert diff, "prompts differ in one token"
    R = torch.tensor([[float("nan") if v is None else v for v in row] for row in ps["restoration"]])
    # patching the whole final-depth state at the readout position restores the clean metric exactly
    assert abs(float(R[D - 1, pos]) - 1.0) < 1e-3
    # patching the embedding of the (only) differing token restores the clean run exactly
    if len(diff) == 1:
        assert abs(float(R[0, diff[0]]) - 1.0) < 1e-3
    with pytest.raises(ServiceError):
        service.patch_sweep(clean["id"], service.create_run("What is 1 + 2?", max_new_tokens=2)["id"])


def test_lazy_level_upgrade_is_deterministic(service):
    r = service.create_run(PROMPT, max_new_tokens=3, trace_level="minimal")
    assert "attn" not in r["tensor_meta"]["keys"]
    t = service.ensure_level(r["id"], "standard")
    assert "attn" in t and "head_z" in t
    rec = service.store.get_run(r["id"])
    assert rec["trace_level"] == "standard"
    assert rec["metrics"]["lazy_recapture"]["max_abs_resid_drift"] == 0.0


def test_persistence_roundtrip(service, run, settings, demo_adapter):
    t1 = service.store.load_tensors(run["id"])
    store2 = ExperimentStore(settings.data_dir, tensor_cache_size=0)
    svc2 = NeuralScopeService(demo_adapter, store2, settings)
    rec = store2.get_run(run["id"])
    assert rec["input_ids"] == run["input_ids"] and rec["target"] == run["target"]
    t2 = store2.load_tensors(run["id"])
    assert torch.equal(t1["resid"], t2["resid"])
    exp = store2.save_experiment("carry test", [run["id"]], notes="first look", payload={"selected": [3, 7]})
    again = svc2.store.get_experiment(exp["id"])
    assert again["name"] == "carry test" and again["run_ids"] == [run["id"]] and again["payload"]["selected"] == [3, 7]
    assert (settings.data_dir / "experiments" / f"{exp['id']}.json").exists()
    assert store2.delete_experiment(exp["id"]) and store2.get_experiment(exp["id"]) is None
    assert store2.delete_run(run["id"]) and store2.get_run(run["id"]) is None


def test_prompt_limits(service):
    with pytest.raises(ServiceError):
        service.create_run("   ")
    with pytest.raises(ServiceError):
        service.create_run("word " * 400)


def test_interventions_compose_against_root_baseline(service, run):
    first = service.run_intervention(run["id"], [HeadAblation(layer=2, head=1)])
    second = service.run_intervention(first["run"]["id"], [HeadAblation(layer=3, head=0)])
    child = service.store.get_run(second["run"]["id"])
    assert child["parent_run_id"] == run["id"]  # lineage points at the root baseline
    assert [i["layer"] for i in child["interventions"]] == [2, 3]
    assert second["comparison"]["composed_from"] == first["run"]["id"]
    # "before" numbers are the root baseline's, identical to the first intervention's "before"
    assert second["comparison"]["before"] == first["comparison"]["before"]
    # identical to applying both edits directly to the baseline
    both = service.run_intervention(run["id"], [HeadAblation(layer=2, head=1), HeadAblation(layer=3, head=0)])
    assert both["comparison"]["after"] == second["comparison"]["after"]


def test_matrix_value_transform(service, run):
    assert service.matrix(run["id"], "target_rank")["value_transform"] == "log10"
    m = service.matrix(run["id"], "resid_norm")
    assert m["value_transform"] == "none" and m["scale"] == "log"
