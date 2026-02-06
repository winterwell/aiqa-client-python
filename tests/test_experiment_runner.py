"""Tests for experiment_runner (score key uses metric.id)."""
import pytest
from unittest.mock import AsyncMock
from aiqa.experiment_runner import _metric_score_key
from aiqa.experiment_runner import ExperimentRunner


def test_metric_score_key_prefers_id():
    """Scores are keyed by metric.id so server/webapp lookup by id works."""
    assert _metric_score_key({"id": "m1", "name": "Accuracy"}) == "m1"
    assert _metric_score_key({"id": "m2", "name": "Relevance"}) == "m2"


def test_metric_score_key_fallback_to_name():
    """When id is missing, fall back to name for backwards compatibility."""
    assert _metric_score_key({"name": "Accuracy"}) == "Accuracy"
    assert _metric_score_key({"id": None, "name": "F1"}) == "F1"


def test_metric_score_key_empty():
    """Empty or missing id/name returns empty string."""
    assert _metric_score_key({}) == ""
    assert _metric_score_key({"id": "", "name": ""}) == ""


@pytest.mark.asyncio
async def test_run_example_sets_aiqa_experiment_attributes(monkeypatch):
    """RunExample should tag root span so server can backfill token stats to the experiment."""
    runner = ExperimentRunner("dataset-1")
    runner.experiment_id = "exp-1"
    runner.experiment = {"parameters": {}}
    runner._dataset_cache = {"metrics": []}
    runner.score_and_store = AsyncMock(return_value={"example": "ex-1", "scores": {"duration": 1}})

    calls = []

    def _record_attr(name, value):
        calls.append((name, value))
        return True

    monkeypatch.setattr("aiqa.experiment_runner.set_span_attribute", _record_attr)
    monkeypatch.setattr("aiqa.experiment_runner.get_active_trace_id", lambda: "trace-1")
    monkeypatch.setattr("aiqa.experiment_runner.flush_tracing", AsyncMock())

    async def call_my_code(input_data, parameters):
        return {"ok": True}

    await runner.run_example({"id": "ex-1", "input": "hello", "metrics": []}, call_my_code)

    assert ("aiqa.experiment", "exp-1") in calls
    assert ("aiqa.example", "ex-1") in calls


@pytest.mark.asyncio
async def test_run_some_examples_filters_by_tag_and_limit(monkeypatch):
    runner = ExperimentRunner("dataset-1")
    runner.get_examples_for_dataset = lambda: [
        {"id": "e1", "tags": ["a", "b"]},
        {"id": "e2", "tags": ["b"]},
        {"id": "e3", "tags": ["a"]},
    ]
    runner.run_example = AsyncMock(return_value=[{"example": "x", "scores": {}}])

    async def call_my_code(input_data, parameters):
        return {"ok": True}

    run_count = await runner.run_some_examples(call_my_code, tag="a", limit=1)

    assert run_count == 1
    runner.run_example.assert_awaited_once()
    called_example = runner.run_example.await_args.args[0]
    assert called_example["id"] == "e1"


@pytest.mark.asyncio
async def test_run_some_examples_resumes_and_skips_existing_results(monkeypatch):
    runner = ExperimentRunner("dataset-1", experiment_id="exp-1")
    runner.get_examples_for_dataset = lambda: [
        {"id": "e1"},
        {"id": "e2"},
    ]
    runner.get_experiment = lambda: {
        "id": "exp-1",
        "results": [{"example": "e1", "scores": {"duration": 1}}],
    }
    runner.experiment = None
    runner.run_example = AsyncMock(return_value=[{"example": "e2", "scores": {}}])

    async def call_my_code(input_data, parameters):
        return {"ok": True}

    run_count = await runner.run_some_examples(call_my_code)

    assert run_count == 1
    runner.run_example.assert_awaited_once()
    called_example = runner.run_example.await_args.args[0]
    assert called_example["id"] == "e2"
