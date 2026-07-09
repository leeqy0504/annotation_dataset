import json
from pathlib import Path

import pytest
import yaml

from pipeline.config import InputConfig, PipelineConfig, Sam2Config
from pipeline.stages.base import StageError
from pipeline.stages.context import DataContext, RunContext, StageContext
from pipeline.stages.training import DatasetPrepareStage, ModelTrainStage

ROOT = Path(__file__).resolve().parents[1]


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


def test_dataset_prepare_stage_requires_copied_images_for_training_handoff(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    export_dir = tmp_path / "run" / "detection_dataset_export"
    output_dir = tmp_path / "run" / "dataset_prepare"
    _write_split(export_dir, "train", "000000.png")
    _write_split(export_dir, "valid", "000001.png")
    config = _config(task_dir)
    config.detection_dataset.copy_images = False
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

    with pytest.raises(
        StageError,
        match="dataset_prepare requires detection_dataset.copy_images=true for training handoff",
    ):
        DatasetPrepareStage().run(config, output_dir, context=context)


class FakeRunner:
    def __init__(self):
        self.config = None

    def train(self, config):
        self.config = config
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


def test_model_train_stage_accepts_rf_detr_alias_before_get_runner(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training_name = "rfdetr_seg_nano"
    config.training = {
        "framework": "rf-detr",
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
    frameworks = []

    def record_get_runner(framework):
        frameworks.append(framework)
        return FakeRunner()

    monkeypatch.setattr("pipeline.stages.training.get_runner", record_get_runner)

    result = ModelTrainStage().run(config, output_dir, context=context)

    assert result == output_dir
    assert frameworks == ["rf-detr"]
    train_result = json.loads((output_dir / "train_result.json").read_text(encoding="utf-8"))
    assert train_result["framework"] == "rf-detr"
    assert train_result["best_weights"].endswith("checkpoint_best_ema.pth")


def test_model_train_stage_scopes_default_relative_training_output_dir(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training_name = "rfdetr_seg_nano"
    config.training = yaml.safe_load((ROOT / "configs" / "training" / "rfdetr_seg_nano.yaml").read_text(encoding="utf-8"))
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
    runner = FakeRunner()
    monkeypatch.setattr("pipeline.stages.training.get_runner", lambda framework: runner)

    ModelTrainStage().run(config, output_dir, context=context)

    resolved = yaml.safe_load((output_dir / "resolved_unitrain_config.yaml").read_text(encoding="utf-8"))
    scoped_output_dir = output_dir / "outputs"
    assert resolved["train"]["output_dir"] == str(scoped_output_dir)
    assert runner.config["train"]["output_dir"] == str(scoped_output_dir)
    assert Path(resolved["train"]["output_dir"]).is_absolute()
    assert resolved["train"]["output_dir"] != "outputs"


def test_model_train_stage_defaults_missing_training_output_dir_to_unitrain(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training_name = "rfdetr_seg_nano"
    config.training = {
        "framework": "rfdetr",
        "model": "seg-nano",
        "task": "segment",
        "data": {"format": "coco"},
        "train": {"epochs": 1, "batch": 1, "device": "cpu"},
        "export": {"format": "onnx"},
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
    runner = FakeRunner()
    monkeypatch.setattr("pipeline.stages.training.get_runner", lambda framework: runner)

    ModelTrainStage().run(config, output_dir, context=context)

    resolved = yaml.safe_load((output_dir / "resolved_unitrain_config.yaml").read_text(encoding="utf-8"))
    default_output_dir = output_dir / "unitrain"
    assert resolved["train"]["output_dir"] == str(default_output_dir)
    assert runner.config["train"]["output_dir"] == str(default_output_dir)


def test_model_train_stage_rejects_unknown_framework_before_get_runner(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training = {
        "framework": "mmdet",
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
        raise AssertionError(f"get_runner should not be called for {framework}")

    monkeypatch.setattr("pipeline.stages.training.get_runner", fail_get_runner)

    with pytest.raises(
        StageError,
        match="model_train supports frameworks 'rfdetr' and 'ultralytics'; got 'mmdet'",
    ):
        ModelTrainStage().run(config, output_dir, context=context)


def test_model_train_stage_converts_coco_to_yolo_for_ultralytics(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "mouse_001"
    dataset_root = tmp_path / "run" / "detection_dataset_export"
    dataset_prepare_dir = tmp_path / "run" / "dataset_prepare"
    output_dir = tmp_path / "run" / "model_train"
    _write_dataset_manifest(dataset_prepare_dir, dataset_root)
    config = _config(task_dir)
    config.training_name = "yolo11n_seg"
    config.training = {
        "framework": "ultralytics",
        "model": "yolo11n-seg",
        "task": "segment",
        "data": {"format": "coco"},
        "train": {"epochs": 1, "batch": 1, "device": "cpu"},
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
    converted = []

    def fake_convert(coco_root, yolo_root, task="detect", framework="yolo"):
        converted.append({
            "coco_root": Path(coco_root),
            "yolo_root": Path(yolo_root),
            "task": task,
            "framework": framework,
        })
        Path(yolo_root).mkdir(parents=True)
        (Path(yolo_root) / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
        return {"nc": 1, "names": ["object"]}

    runner = FakeRunner()
    monkeypatch.setattr("pipeline.stages.training.convert_coco_dataset", fake_convert, raising=False)
    monkeypatch.setattr("pipeline.stages.training.get_runner", lambda framework: runner)

    ModelTrainStage().run(config, output_dir, context=context)

    yolo_root = output_dir / "dataset_yolo"
    assert converted == [{
        "coco_root": dataset_root,
        "yolo_root": yolo_root,
        "task": "segment",
        "framework": "ultralytics",
    }]
    assert runner.config["data"]["path"] == str(yolo_root)
    assert runner.config["data"]["format"] == "yolo"
    assert runner.config["data_yaml"] == str(yolo_root / "data.yaml")
    assert runner.config["data"]["nc"] == 1
    assert runner.config["data"]["names"] == ["object"]
    resolved = yaml.safe_load((output_dir / "resolved_unitrain_config.yaml").read_text(encoding="utf-8"))
    assert resolved["data"]["path"] == str(yolo_root)
    assert resolved["data"]["format"] == "yolo"


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


def test_model_train_stage_rejects_non_mapping_train_config(tmp_path):
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
        "data": {"format": "coco"},
        "train": "outputs",
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

    with pytest.raises(StageError, match="Training config field 'train' must be a mapping"):
        ModelTrainStage().run(config, output_dir, context=context)
