import json
from pathlib import Path

import pytest
import yaml

from pipeline.config import InputConfig, PipelineConfig, Sam2Config
from pipeline.stages.base import StageError
from pipeline.stages.context import DataContext, RunContext, StageContext
from pipeline.stages.training import DatasetPrepareStage, ModelTrainStage


def _config(task_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        task="mouse_001",
        preset="annotation_to_unitrain",
        input=InputConfig(rgbd_dir=str(task_dir)),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
    )


def _write_split(root: Path, split: str, image_name: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    (split_dir / image_name).write_bytes(b"fake image")
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps({
            "images": [{"id": 1, "file_name": image_name, "width": 4, "height": 3}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
            "categories": [{"id": 0, "name": "object", "supercategory": "object"}],
        }),
        encoding="utf-8",
    )


def _write_dataset_manifest(dataset_prepare_dir: Path, dataset_root: Path) -> None:
    dataset_prepare_dir.mkdir(parents=True)
    (dataset_prepare_dir / "dataset_manifest.json").write_text(
        json.dumps({
            "dataset_id": "mouse_001:run42:detection_dataset_export",
            "task": "mouse_001",
            "run_id": "run42",
            "source_stage": "detection_dataset_export",
            "root": str(dataset_root),
            "format": "coco",
        }),
        encoding="utf-8",
    )


def test_dataset_prepare_stage_writes_manifest_from_export_output(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    export_dir = tmp_path / "run" / "detection_dataset_export"
    output_dir = tmp_path / "run" / "dataset_prepare"
    _write_split(export_dir, "train", "000000.png")
    _write_split(export_dir, "valid", "000001.png")
    context = StageContext(
        run=RunContext(run_id="run42", task_name="mouse_001"),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "run",
            output_dir=output_dir,
            inputs={"detection_dataset_export": export_dir},
        ),
        stage_name="dataset_prepare",
    )

    result = DatasetPrepareStage().run(_config(task_dir), output_dir, context=context)

    assert result == output_dir
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "mouse_001:run42:detection_dataset_export"
    assert manifest["root"] == str(export_dir)


class FakeRunner:
    def train(self, config):
        output_dir = Path(config["train"]["output_dir"])
        train_output_dir = output_dir / "rfdetr_fake"
        train_output_dir.mkdir(parents=True)
        best_weights = train_output_dir / "checkpoint_best_ema.pth"
        best_weights.write_bytes(b"weights")
        return {
            "output_dir": str(train_output_dir),
            "best_weights": str(best_weights),
        }


def test_model_train_stage_writes_resolved_config_and_train_result(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_split(dataset_root, "train", "000000.png")
    _write_split(dataset_root, "valid", "000001.png")
    DatasetPrepareStage().run(
        _config(task_dir),
        dataset_prepare_dir,
        context=StageContext(
            run=RunContext(run_id="run42", task_name="mouse_001"),
            data=DataContext(
                task_dir=task_dir,
                run_dir=tmp_path / "run",
                output_dir=dataset_prepare_dir,
                inputs={"detection_dataset_export": dataset_root},
            ),
            stage_name="dataset_prepare",
        ),
    )
    config = _config(task_dir)
    config.training_name = "rfdetr_seg_nano"
    config.training = {
        "framework": "rfdetr",
        "model": "seg-nano",
        "task": "segment",
        "data": {"format": "coco"},
        "train": {"epochs": 1, "batch": 1, "device": "cpu", "output_dir": str(tmp_path / "train_outputs")},
        "export": {"format": "onnx"},
    }
    context = StageContext(
        run=RunContext(run_id="run42", task_name="mouse_001", metadata={}),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "run",
            output_dir=output_dir,
            inputs={"dataset_prepare": dataset_prepare_dir},
        ),
        stage_name="model_train",
    )
    monkeypatch.setattr("pipeline.stages.training.get_runner", lambda framework: FakeRunner())

    result = ModelTrainStage().run(config, output_dir, context=context)

    assert result == output_dir
    resolved = yaml.safe_load((output_dir / "resolved_unitrain_config.yaml").read_text(encoding="utf-8"))
    assert resolved["data"]["path"] == str(dataset_root)
    assert resolved["data"]["format"] == "coco"
    assert resolved["train"]["output_dir"] == str(tmp_path / "train_outputs")
    train_result = json.loads((output_dir / "train_result.json").read_text(encoding="utf-8"))
    assert train_result["framework"] == "rfdetr"
    assert train_result["model"] == "seg-nano"
    assert train_result["task"] == "segment"
    assert train_result["best_weights"].endswith("checkpoint_best_ema.pth")
    assert context.metadata["model_train"]["best_weights"] == train_result["best_weights"]


def test_model_train_stage_wraps_get_runner_failures(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training = {
        "framework": "nope",
        "model": "seg-nano",
        "task": "segment",
        "data": {"format": "coco"},
    }
    context = StageContext(
        run=RunContext(run_id="run42", task_name="mouse_001"),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "run",
            output_dir=output_dir,
            inputs={"dataset_prepare": dataset_prepare_dir},
        ),
        stage_name="model_train",
    )

    def fail_get_runner(framework):
        raise ValueError(f"unsupported framework: {framework}")

    monkeypatch.setattr("pipeline.stages.training.get_runner", fail_get_runner)

    with pytest.raises(StageError, match="Training failed: unsupported framework: nope"):
        ModelTrainStage().run(config, output_dir, context=context)


def test_model_train_stage_rejects_non_mapping_data_config(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training = {
        "framework": "rfdetr",
        "model": "seg-nano",
        "task": "segment",
        "data": "coco",
    }
    context = StageContext(
        run=RunContext(run_id="run42", task_name="mouse_001"),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "run",
            output_dir=output_dir,
            inputs={"dataset_prepare": dataset_prepare_dir},
        ),
        stage_name="model_train",
    )

    with pytest.raises(StageError, match="Training config field 'data' must be a mapping"):
        ModelTrainStage().run(config, output_dir, context=context)
