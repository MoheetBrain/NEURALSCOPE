"""Shared fixtures.

Fast tests use the DEMO fixture model (tiny random-weight Qwen2 architecture,
real Qwen tokenizer) so every instrumentation code path runs in milliseconds.
Tests marked ``slow`` load a real pretrained model (default Qwen2.5-0.5B-Instruct,
same architecture family as the V1 model; override with NEURALSCOPE_TEST_MODEL).
"""

from __future__ import annotations

import os

import pytest
import torch

from neuralscope.config import DEMO_MODEL_ID, Settings
from neuralscope.experiments.service import NeuralScopeService
from neuralscope.experiments.store import ExperimentStore

DEMO_TOKENIZER = os.environ.get("NEURALSCOPE_DEMO_TOKENIZER", "Qwen/Qwen2.5-0.5B-Instruct")
REAL_TEST_MODEL = os.environ.get("NEURALSCOPE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def demo_adapter():
    from neuralscope.instrumentation.demo import build_demo_adapter

    return build_demo_adapter(DEMO_TOKENIZER)


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings()
    s.model_id = DEMO_MODEL_ID
    s.data_dir = tmp_path / "data"
    s.trace_level = "standard"
    return s


@pytest.fixture()
def store(settings) -> ExperimentStore:
    return ExperimentStore(settings.data_dir)


@pytest.fixture()
def service(demo_adapter, store, settings) -> NeuralScopeService:
    return NeuralScopeService(demo_adapter, store, settings)


@pytest.fixture(scope="session")
def real_adapter():
    from neuralscope.instrumentation.hf_decoder import HFDecoderAdapter

    torch.set_num_threads(max(1, torch.get_num_threads()))
    a = HFDecoderAdapter(REAL_TEST_MODEL, device="cpu", dtype="float32")
    a.load()
    return a


PROMPT = "What is 37 + 58? Give only the answer."
