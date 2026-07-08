from pathlib import Path

import pytest

from pipeline.config import InputConfig, PipelineConfig, Sam2Config
from pipeline.manifest import Manifest
from pipeline.pipeline import PipelineOrchestrator
from pipeline.stages.context import StageContext


class FakeStage:
    def __init__(self, name: str):
        self.name = name

    def run(self, config, output_dir, context=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.name == "model_train":
            context.metadata["model_train"] = {
                "framework": "rfdetr",
                "model": "seg-nano",
                "task": "segment",
                "train_output_dir": str(output_dir / "train"),
                "best_weights": str(output_dir / "train" / "checkpoint_best_ema.pth"),
                "resolved_config": str(output_dir / "resolved_unitrain_config.yaml"),
            }
        if self.name == "metadata_mutator":
            context.metadata["model_train"]["best_weights"] = "rewritten-later.pth"
        return output_dir


class FailingStage:
    def run(self, config, output_dir, context=None):
        raise RuntimeError("boom")


def test_orchestrator_persists_model_train_metadata(tmp_path, monkeypatch):
    config = PipelineConfig(
        task="mouse_001",
        preset="annotation_to_unitrain",
        input=InputConfig(rgbd_dir=str(tmp_path / "tasks" / "mouse_001")),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
        output_dir=str(tmp_path / "output"),
        pipeline_stages=["dataset_prepare", "model_train"],
    )
    monkeypatch.setattr("pipeline.pipeline.get_stage", lambda name: FakeStage(name))

    PipelineOrchestrator().run_preset(config, force=True)

    manifest = Manifest.load(str(tmp_path / "output" / "mouse_001" / "manifest.json"))
    assert manifest.stages["model_train"]["metadata"]["best_weights"].endswith(
        "checkpoint_best_ema.pth"
    )
    assert manifest.metadata["model_train"]["best_weights"].endswith(
        "checkpoint_best_ema.pth"
    )


def test_completed_stage_metadata_is_snapshotted(tmp_path, monkeypatch):
    config = PipelineConfig(
        task="mouse_001",
        preset="annotation_to_unitrain",
        input=InputConfig(rgbd_dir=str(tmp_path / "tasks" / "mouse_001")),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
        output_dir=str(tmp_path / "output"),
        pipeline_stages=["model_train", "metadata_mutator"],
    )
    monkeypatch.setattr("pipeline.pipeline.get_stage", lambda name: FakeStage(name))

    PipelineOrchestrator().run_preset(config, force=True)

    manifest = Manifest.load(str(tmp_path / "output" / "mouse_001" / "manifest.json"))
    assert manifest.stages["model_train"]["metadata"]["best_weights"].endswith(
        "checkpoint_best_ema.pth"
    )
    assert manifest.metadata["model_train"]["best_weights"] == "rewritten-later.pth"


def test_base_context_metadata_is_not_persisted(tmp_path, monkeypatch):
    config = PipelineConfig(
        task="mouse_001",
        preset="annotation_to_unitrain",
        input=InputConfig(rgbd_dir=str(tmp_path / "tasks" / "mouse_001")),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
        output_dir=str(tmp_path / "output"),
        pipeline_stages=["dataset_prepare"],
    )
    monkeypatch.setattr("pipeline.pipeline.get_stage", lambda name: FakeStage(name))

    PipelineOrchestrator().run_preset(
        config,
        force=True,
        context=StageContext(metadata={"transient": object()}),
    )

    manifest = Manifest.load(str(tmp_path / "output" / "mouse_001" / "manifest.json"))
    assert "transient" not in manifest.metadata


def test_run_stage_persists_model_train_metadata(tmp_path, monkeypatch):
    config = PipelineConfig(
        task="mouse_001",
        preset="annotation_to_unitrain",
        input=InputConfig(rgbd_dir=str(tmp_path / "tasks" / "mouse_001")),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
        output_dir=str(tmp_path / "output"),
    )
    monkeypatch.setattr("pipeline.pipeline.get_stage", lambda name: FakeStage(name))

    PipelineOrchestrator().run_stage(config, "model_train", force=True)

    manifest = Manifest.load(str(tmp_path / "output" / "mouse_001" / "manifest.json"))
    assert manifest.stages["model_train"]["metadata"]["best_weights"].endswith(
        "checkpoint_best_ema.pth"
    )
    assert manifest.metadata["model_train"]["best_weights"].endswith(
        "checkpoint_best_ema.pth"
    )


def test_failed_forced_stage_rerun_clears_stale_run_metadata(tmp_path, monkeypatch):
    config = PipelineConfig(
        task="mouse_001",
        preset="annotation_to_unitrain",
        input=InputConfig(rgbd_dir=str(tmp_path / "tasks" / "mouse_001")),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
        output_dir=str(tmp_path / "output"),
    )
    monkeypatch.setattr("pipeline.pipeline.get_stage", lambda name: FakeStage(name))

    PipelineOrchestrator().run_stage(config, "model_train", force=True)

    manifest_path = tmp_path / "output" / "mouse_001" / "manifest.json"
    manifest = Manifest.load(str(manifest_path))
    assert "model_train" in manifest.metadata

    monkeypatch.setattr("pipeline.pipeline.get_stage", lambda name: FailingStage())

    with pytest.raises(RuntimeError, match="boom"):
        PipelineOrchestrator().run_stage(config, "model_train", force=True)

    manifest = Manifest.load(str(manifest_path))
    assert manifest.stages["model_train"]["status"] == "failed"
    assert "model_train" not in manifest.metadata
