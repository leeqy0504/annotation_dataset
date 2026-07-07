# Annotation to UniTrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MVP CLI-driven end-to-end pipeline from annotation dataset export to UniTrain model weights inside the unified `annotation-dataset-pipeline` project.

**Architecture:** Keep `pipeline/` as the orchestration layer, migrate UniTrain into the same project root, add a `datasets/` boundary package, and implement `dataset_prepare` plus `model_train` as normal registered stages. The dataset stage writes a stable manifest, and the training stage consumes that manifest, writes a resolved UniTrain config, calls the runner abstraction, and records weight outputs.

**Tech Stack:** Python 3.10+, dataclasses, YAML via PyYAML, pytest, existing pipeline stage registry, existing UniTrain runner abstractions.

---

## File Structure

Create or modify these files:

- Modify `pyproject.toml`: include migrated `unitrain*`, `cli*`, and new `datasets*` packages; expose migrated UniTrain console scripts.
- Create `configs/pipelines/annotation_to_unitrain.yaml`: full end-to-end stage sequence.
- Create `configs/training/rfdetr_seg_nano.yaml`: default RF-DETR segmentation preset.
- Modify `pipeline/config.py`: add training config loading, `training_overrides` merge, `PipelineConfig.training_name`, and `PipelineConfig.training`.
- Modify `pipeline/pipeline.py`: include training config in `resolved_config.yaml`, pass manifest metadata through `StageContext`, and persist metadata written by stages.
- Modify `pipeline/manifest.py`: allow stage entries to carry metadata through `mark_stage_done`.
- Modify `pipeline/stages/__init__.py`: import the new training stages module.
- Create `pipeline/stages/training.py`: register `dataset_prepare` and `model_train`.
- Create `datasets/__init__.py`: public dataset helper exports.
- Create `datasets/manifest.py`: validate Roboflow-style COCO and write `dataset_manifest.json`.
- Copy from `../unitrain-dev/unitrain/` to `unitrain/`: migrated training core.
- Copy from `../unitrain-dev/cli/` to `cli/`: migrated train/eval/export/predict entry modules.
- Copy from `../unitrain-dev/envs/` to `envs/`: framework dependency files.
- Copy from `../unitrain-dev/weights/` to `weights/`: UniTrain pretrained-weight directory convention.
- Copy `../unitrain-dev/run.sh` to `run_unitrain.sh`: optional direct UniTrain helper inside the unified root.
- Modify `tests/test_standalone_layout.py`: training config and stage registry coverage.
- Create `tests/test_dataset_manifest.py`: dataset manifest validation tests.
- Create `tests/test_training_stages.py`: `dataset_prepare` and `model_train` stage tests with mocked runner.
- Create `tests/test_annotation_to_unitrain_orchestration.py`: end-to-end preset ordering and manifest metadata test.

The migrated `unitrain-dev/unitrain/runners/_scripts/rfdetr_eval.py` has local user changes. Copy the working tree version, not the last committed version.

---

## Task 1: Migrate UniTrain Into the Unified Project Root

**Files:**
- Modify: `pyproject.toml`
- Create: `unitrain/`
- Create: `cli/`
- Create: `envs/`
- Create: `weights/`
- Create: `run_unitrain.sh`
- Test: `tests/test_unitrain_migration.py`

- [ ] **Step 1: Write failing migration smoke tests**

Create `tests/test_unitrain_migration.py`:

```python
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unitrain_package_imports_after_migration():
    unitrain = importlib.import_module("unitrain")

    assert hasattr(unitrain, "get_runner")
    assert hasattr(unitrain, "load_config")


def test_unitrain_cli_modules_import_after_migration():
    train = importlib.import_module("cli.train")
    evaluate = importlib.import_module("cli.eval")
    export = importlib.import_module("cli.export")
    predict = importlib.import_module("cli.predict")

    assert callable(train.main)
    assert callable(evaluate.main)
    assert callable(export.main)
    assert callable(predict.main)


def test_unitrain_support_files_are_inside_unified_root():
    assert (ROOT / "envs" / "rfdetr.txt").exists()
    assert (ROOT / "envs" / "ultralytics.txt").exists()
    assert (ROOT / "weights").is_dir()
    assert (ROOT / "run_unitrain.sh").exists()
```

- [ ] **Step 2: Run the failing smoke tests**

Run:

```bash
pytest tests/test_unitrain_migration.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'unitrain'`.

- [ ] **Step 3: Copy UniTrain source and support files**

Run from `/Users/lee/code/uni-platform/annotation-dataset-pipeline`:

```bash
cp -R ../unitrain-dev/unitrain ./unitrain
cp -R ../unitrain-dev/cli ./cli
cp -R ../unitrain-dev/envs ./envs
cp -R ../unitrain-dev/weights ./weights
cp ../unitrain-dev/run.sh ./run_unitrain.sh
chmod +x ./run_unitrain.sh
```

Then verify the user-modified RF-DETR eval file came across:

```bash
diff -u ../unitrain-dev/unitrain/runners/_scripts/rfdetr_eval.py unitrain/runners/_scripts/rfdetr_eval.py
```

Expected: no diff output.

- [ ] **Step 4: Modify package discovery and scripts**

Edit `pyproject.toml` so the package sections become:

```toml
[project.scripts]
annotation-dataset = "pipeline.cli:main"
annotation_dataset = "pipeline.cli:main"
pipeline = "pipeline.cli:main"
unitrain-train = "cli.train:main"
unitrain-predict = "cli.predict:main"
unitrain-export = "cli.export:main"
unitrain-eval = "cli.eval:main"

[tool.setuptools.packages.find]
include = ["pipeline*", "datasets*", "unitrain*", "cli*"]
exclude = ["configs*", "registry*", "tasks*", "tests*", "tools*"]
```

- [ ] **Step 5: Run the migration smoke tests**

Run:

```bash
pytest tests/test_unitrain_migration.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 6: Commit migration**

Run:

```bash
git add pyproject.toml unitrain cli envs weights run_unitrain.sh tests/test_unitrain_migration.py
git commit -m "Migrate UniTrain into annotation pipeline root"
```

---

## Task 2: Add Training Preset Config Loading

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/pipeline.py`
- Create: `configs/pipelines/annotation_to_unitrain.yaml`
- Create: `configs/training/rfdetr_seg_nano.yaml`
- Modify: `tests/test_standalone_layout.py`

- [ ] **Step 1: Write failing config tests**

Append these tests to `tests/test_standalone_layout.py`:

```python
def test_annotation_to_unitrain_config_loads_training_preset_and_overrides(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: mouse_001
pipeline: annotation_to_unitrain
runtime: server
class_id: 0
input:
  rgbd_dir: ./tasks/mouse_001/
sam2:
  points: [[10, 20]]
  labels: [1]
detection_dataset:
  train_ratio: 0.7
training: rfdetr_seg_nano
training_overrides:
  train:
    epochs: 20
    batch: 2
    device: "cpu"
output_dir: output/
""",
        encoding="utf-8",
    )

    config = load_config(str(task_dir / "task.yaml"), project_root=ROOT)

    assert config.preset == "annotation_to_unitrain"
    assert config.pipeline_stages == [
        "prompt_mask",
        "sam2_video_propagation",
        "mask_qa",
        "review_pack",
        "detection_dataset_export",
        "dataset_prepare",
        "model_train",
    ]
    assert config.training_name == "rfdetr_seg_nano"
    assert config.training["framework"] == "rfdetr"
    assert config.training["model"] == "seg-nano"
    assert config.training["task"] == "segment"
    assert config.training["data"]["format"] == "coco"
    assert config.training["train"]["epochs"] == 20
    assert config.training["train"]["batch"] == 2
    assert config.training["train"]["device"] == "cpu"


def test_training_overrides_must_be_mapping(tmp_path):
    task_dir = tmp_path / "tasks" / "bad_training"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: bad_training
pipeline: annotation_to_unitrain
runtime: server
input:
  rgbd_dir: ./tasks/bad_training/
sam2:
  points: [[1, 2]]
  labels: [1]
training: rfdetr_seg_nano
training_overrides: true
""",
        encoding="utf-8",
    )

    try:
        load_config(str(task_dir / "task.yaml"), project_root=ROOT)
    except Exception as exc:
        assert "training_overrides must be a mapping" in str(exc)
    else:
        raise AssertionError("load_config should reject non-mapping training_overrides")
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
pytest tests/test_standalone_layout.py::test_annotation_to_unitrain_config_loads_training_preset_and_overrides tests/test_standalone_layout.py::test_training_overrides_must_be_mapping -v
```

Expected: FAIL because `PipelineConfig` has no `training_name` field and the preset files do not exist.

- [ ] **Step 3: Add pipeline and training preset YAML files**

Create `configs/pipelines/annotation_to_unitrain.yaml`:

```yaml
preset: annotation_to_unitrain
stages:
  - prompt_mask
  - sam2_video_propagation
  - mask_qa
  - review_pack
  - detection_dataset_export
  - dataset_prepare
  - model_train
```

Create `configs/training/rfdetr_seg_nano.yaml`:

```yaml
framework: rfdetr
model: seg-nano
task: segment
data:
  format: coco
train:
  epochs: 100
  batch: 4
  device: 0
  output_dir: outputs
export:
  format: onnx
```

- [ ] **Step 4: Add training fields and merge logic**

In `pipeline/config.py`, add fields to `PipelineConfig`:

```python
    training_name: str | None = None
    training: dict[str, Any] = field(default_factory=dict)
```

Add this helper below `_load_named_yaml`:

```python
def _load_training_config(project_root: Path, raw: dict) -> tuple[str | None, dict]:
    training_name = raw.get("training")
    if not training_name:
        return None, {}

    training_data = _load_named_yaml(project_root / "configs" / "training", training_name)
    overrides = raw.get("training_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ConfigError("training_overrides must be a mapping")
    return str(training_name), _deep_merge(training_data, overrides)
```

In `_load_layered_config`, after runtime config is merged and before returning, add:

```python
    training_name, training_config = _load_training_config(project_root, raw)
    if training_name:
        merged["training_name"] = training_name
        merged["training"] = training_config
```

In `load_config`, pass the new fields to `PipelineConfig`:

```python
        training_name=resolved.get("training_name"),
        training=resolved.get("training", {}),
```

- [ ] **Step 5: Write training into resolved pipeline config**

In `pipeline/pipeline.py`, add these keys to `_write_resolved_config`:

```python
            "training_name": config.training_name,
            "training": config.training,
```

- [ ] **Step 6: Run config tests**

Run:

```bash
pytest tests/test_standalone_layout.py::test_annotation_to_unitrain_config_loads_training_preset_and_overrides tests/test_standalone_layout.py::test_training_overrides_must_be_mapping -v
```

Expected: PASS.

- [ ] **Step 7: Run existing layout tests**

Run:

```bash
pytest tests/test_standalone_layout.py -v
```

Expected: PASS for all tests that do not depend on new stage registration.

- [ ] **Step 8: Commit config layer**

Run:

```bash
git add pipeline/config.py pipeline/pipeline.py configs/pipelines/annotation_to_unitrain.yaml configs/training/rfdetr_seg_nano.yaml tests/test_standalone_layout.py
git commit -m "Add training preset config layer"
```

---

## Task 3: Add Dataset Manifest Validation Package

**Files:**
- Create: `datasets/__init__.py`
- Create: `datasets/manifest.py`
- Create: `tests/test_dataset_manifest.py`

- [ ] **Step 1: Write failing dataset manifest tests**

Create `tests/test_dataset_manifest.py`:

```python
import json
from pathlib import Path

from datasets.manifest import DatasetValidationError, prepare_dataset_manifest


def _write_split(root: Path, split: str, image_names: list[str], categories: list[dict]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    images = []
    annotations = []
    for index, image_name in enumerate(image_names, start=1):
        (split_dir / image_name).write_bytes(b"fake image")
        images.append({
            "id": index,
            "file_name": image_name,
            "width": 4,
            "height": 3,
        })
        annotations.append({
            "id": index,
            "image_id": index,
            "category_id": categories[0]["id"],
            "bbox": [0, 0, 2, 2],
            "area": 4,
            "iscrowd": 0,
        })
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps({
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }),
        encoding="utf-8",
    )


def test_prepare_dataset_manifest_writes_roboflow_coco_contract(tmp_path):
    dataset_root = tmp_path / "detection_dataset_export"
    output_dir = tmp_path / "dataset_prepare"
    categories = [{"id": 0, "name": "object", "supercategory": "object"}]
    _write_split(dataset_root, "train", ["000000.png", "000001.png"], categories)
    _write_split(dataset_root, "valid", ["000002.png"], categories)

    manifest = prepare_dataset_manifest(
        dataset_root=dataset_root,
        output_dir=output_dir,
        task_name="mouse_001",
        run_id="run42",
        source_stage="detection_dataset_export",
    )

    manifest_path = output_dir / "dataset_manifest.json"
    assert manifest_path.exists()
    assert manifest["dataset_id"] == "mouse_001:run42:detection_dataset_export"
    assert manifest["format"] == "roboflow_coco"
    assert manifest["root"] == str(dataset_root)
    assert manifest["splits"]["train"]["image_count"] == 2
    assert manifest["splits"]["train"]["annotation_count"] == 2
    assert manifest["splits"]["valid"]["image_count"] == 1
    assert manifest["splits"]["valid"]["annotation_count"] == 1
    assert manifest["categories"] == [{"id": 0, "name": "object"}]
    assert manifest["validation"] == {"status": "passed", "warnings": []}


def test_prepare_dataset_manifest_rejects_missing_referenced_image(tmp_path):
    dataset_root = tmp_path / "detection_dataset_export"
    output_dir = tmp_path / "dataset_prepare"
    categories = [{"id": 0, "name": "object"}]
    _write_split(dataset_root, "train", ["000000.png"], categories)
    _write_split(dataset_root, "valid", ["000001.png"], categories)
    (dataset_root / "valid" / "000001.png").unlink()

    try:
        prepare_dataset_manifest(
            dataset_root=dataset_root,
            output_dir=output_dir,
            task_name="mouse_001",
            run_id=None,
            source_stage="detection_dataset_export",
        )
    except DatasetValidationError as exc:
        assert "Referenced image not found" in str(exc)
        assert "000001.png" in str(exc)
    else:
        raise AssertionError("prepare_dataset_manifest should reject missing images")
```

- [ ] **Step 2: Run dataset tests and verify failure**

Run:

```bash
pytest tests/test_dataset_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'datasets.manifest'`.

- [ ] **Step 3: Create dataset package exports**

Create `datasets/__init__.py`:

```python
"""Dataset validation and manifest helpers."""

from .manifest import DatasetValidationError, prepare_dataset_manifest

__all__ = ["DatasetValidationError", "prepare_dataset_manifest"]
```

- [ ] **Step 4: Implement dataset manifest writer**

Create `datasets/manifest.py`:

```python
"""Dataset manifest generation for pipeline-to-training handoff."""

import json
from pathlib import Path
from typing import Any


class DatasetValidationError(Exception):
    """Raised when an exported dataset does not satisfy the training contract."""


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _category_key(categories: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [(int(item["id"]), str(item["name"])) for item in categories]


def _normalized_categories(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": int(item["id"]), "name": str(item["name"])} for item in categories]


def _validate_split(dataset_root: Path, split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_dir = dataset_root / split_name
    annotations_path = split_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise DatasetValidationError(f"Missing annotation file for split '{split_name}': {annotations_path}")

    coco = _read_json(annotations_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not images:
        raise DatasetValidationError(f"Split '{split_name}' has no images")
    if not categories:
        raise DatasetValidationError(f"Split '{split_name}' has no categories")

    for image in images:
        file_name = image.get("file_name")
        image_path = split_dir / str(file_name)
        if not image_path.exists():
            raise DatasetValidationError(f"Referenced image not found for split '{split_name}': {image_path}")

    split_manifest = {
        "images_dir": str(split_dir),
        "annotations": str(annotations_path),
        "image_count": len(images),
        "annotation_count": len(annotations),
    }
    return split_manifest, categories


def prepare_dataset_manifest(
    *,
    dataset_root: Path,
    output_dir: Path,
    task_name: str,
    run_id: str | None,
    source_stage: str,
) -> dict[str, Any]:
    """Validate Roboflow-style COCO output and write dataset_manifest.json."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    split_manifests: dict[str, dict[str, Any]] = {}
    category_sets: list[list[tuple[int, str]]] = []
    first_categories: list[dict[str, Any]] | None = None
    for split_name in ("train", "valid"):
        split_manifest, categories = _validate_split(dataset_root, split_name)
        split_manifests[split_name] = split_manifest
        category_sets.append(_category_key(categories))
        if first_categories is None:
            first_categories = categories

    if len({tuple(category_set) for category_set in category_sets}) != 1:
        raise DatasetValidationError("Inconsistent categories between train and valid splits")

    dataset_id = f"{task_name}:{run_id or 'default'}:{source_stage}"
    manifest = {
        "dataset_id": dataset_id,
        "format": "roboflow_coco",
        "root": str(dataset_root),
        "splits": split_manifests,
        "categories": _normalized_categories(first_categories or []),
        "source_stage": source_stage,
        "validation": {
            "status": "passed",
            "warnings": [],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
```

- [ ] **Step 5: Run dataset tests**

Run:

```bash
pytest tests/test_dataset_manifest.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit dataset manifest package**

Run:

```bash
git add datasets tests/test_dataset_manifest.py
git commit -m "Add dataset manifest validation"
```

---

## Task 4: Register Dataset Prepare Stage

**Files:**
- Create: `pipeline/stages/training.py`
- Modify: `pipeline/stages/__init__.py`
- Modify: `tests/test_standalone_layout.py`
- Create: `tests/test_training_stages.py`

- [ ] **Step 1: Write failing stage registration and dataset stage tests**

Update `test_only_annotation_dataset_runtime_stages_are_registered` in `tests/test_standalone_layout.py` to:

```python
def test_pipeline_runtime_stages_are_registered():
    assert set(list_stages()) == {
        "masks",
        "prompt_mask",
        "sam2_video_propagation",
        "mask_qa",
        "review_pack",
        "detection_dataset_export",
        "dataset_prepare",
        "model_train",
    }
```

Create the first part of `tests/test_training_stages.py`:

```python
import json
from pathlib import Path

from pipeline.config import InputConfig, PipelineConfig, Sam2Config
from pipeline.stages.context import DataContext, RunContext, StageContext
from pipeline.stages.training import DatasetPrepareStage


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
```

- [ ] **Step 2: Run stage tests and verify failure**

Run:

```bash
pytest tests/test_standalone_layout.py::test_pipeline_runtime_stages_are_registered tests/test_training_stages.py::test_dataset_prepare_stage_writes_manifest_from_export_output -v
```

Expected: FAIL because `pipeline.stages.training` does not exist.

- [ ] **Step 3: Implement `DatasetPrepareStage` and a stage input helper**

Create `pipeline/stages/training.py`:

```python
"""Training-related pipeline stages."""

from pathlib import Path

import yaml

from datasets import DatasetValidationError, prepare_dataset_manifest
from pipeline.config import PipelineConfig
from pipeline.manifest import load_manifest_for_config
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def _stage_input(config: PipelineConfig, context: StageContext | None, stage_name: str) -> Path:
    if context and context.data and context.data.get_input(stage_name):
        return context.input(stage_name)
    manifest = load_manifest_for_config(config)
    output = manifest.get_output_dir(stage_name)
    if output:
        return Path(output)
    raise StageError(f"Required stage input not found: {stage_name}")


@register_stage("dataset_prepare")
class DatasetPrepareStage(BaseStage):
    @property
    def name(self) -> str:
        return "dataset_prepare"

    def run(
        self,
        config: PipelineConfig,
        output_dir: Path,
        context: StageContext | None = None,
    ) -> Path:
        dataset_root = _stage_input(config, context, "detection_dataset_export")
        run_id = context.run.run_id if context and context.run else config.run_id
        try:
            prepare_dataset_manifest(
                dataset_root=dataset_root,
                output_dir=output_dir,
                task_name=config.task,
                run_id=run_id,
                source_stage="detection_dataset_export",
            )
        except DatasetValidationError as exc:
            raise StageError(str(exc)) from exc
        return output_dir
```

The `yaml` import is used by the next task in this same module.

- [ ] **Step 4: Register the module import**

Add this import to `pipeline/stages/__init__.py`:

```python
from pipeline.stages import training  # noqa: E402,F401
```

- [ ] **Step 5: Run dataset stage tests**

Run:

```bash
pytest tests/test_standalone_layout.py::test_pipeline_runtime_stages_are_registered tests/test_training_stages.py::test_dataset_prepare_stage_writes_manifest_from_export_output -v
```

Expected: PASS.

- [ ] **Step 6: Commit dataset stage**

Run:

```bash
git add pipeline/stages/__init__.py pipeline/stages/training.py tests/test_standalone_layout.py tests/test_training_stages.py
git commit -m "Add dataset prepare stage"
```

---

## Task 5: Add Model Train Stage With Mocked Runner

**Files:**
- Modify: `pipeline/stages/training.py`
- Modify: `tests/test_training_stages.py`

- [ ] **Step 1: Append failing model train tests**

Append to `tests/test_training_stages.py`:

```python
import yaml

from pipeline.stages.training import ModelTrainStage


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
```

- [ ] **Step 2: Run model train test and verify failure**

Run:

```bash
pytest tests/test_training_stages.py::test_model_train_stage_writes_resolved_config_and_train_result -v
```

Expected: FAIL because `ModelTrainStage` is not defined.

- [ ] **Step 3: Import JSON and UniTrain runner in training stage module**

Add imports near the top of `pipeline/stages/training.py`:

```python
import copy
import json

from unitrain import get_runner
```

- [ ] **Step 4: Add training config helper functions**

Append these helpers to `pipeline/stages/training.py` before the stage classes:

```python
def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolved_training_config(config: PipelineConfig, dataset_manifest: dict) -> dict:
    if not config.training:
        raise StageError("Training config is required for model_train")
    resolved = copy.deepcopy(config.training)
    data = resolved.setdefault("data", {})
    data["path"] = dataset_manifest["root"]
    data.setdefault("format", dataset_manifest.get("format", "coco").replace("roboflow_", ""))
    return resolved
```

- [ ] **Step 5: Implement `ModelTrainStage`**

Append this class to `pipeline/stages/training.py`:

```python
@register_stage("model_train")
class ModelTrainStage(BaseStage):
    @property
    def name(self) -> str:
        return "model_train"

    def run(
        self,
        config: PipelineConfig,
        output_dir: Path,
        context: StageContext | None = None,
    ) -> Path:
        dataset_prepare_dir = _stage_input(config, context, "dataset_prepare")
        dataset_manifest_path = dataset_prepare_dir / "dataset_manifest.json"
        if not dataset_manifest_path.exists():
            raise StageError(f"Dataset manifest not found: {dataset_manifest_path}")

        dataset_manifest = _read_json(dataset_manifest_path)
        resolved_config = _resolved_training_config(config, dataset_manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = output_dir / "resolved_unitrain_config.yaml"
        resolved_config_path.write_text(
            yaml.dump(resolved_config, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        framework = resolved_config.get("framework")
        if not framework:
            raise StageError("Training config missing required field: framework")
        runner = get_runner(str(framework))
        try:
            train_info = runner.train(resolved_config)
        except Exception as exc:
            raise StageError(f"Training failed: {exc}") from exc

        if not train_info:
            raise StageError("Training did not return output information")
        train_output_dir = train_info.get("output_dir", "")
        best_weights = train_info.get("best_weights", "")
        if not train_output_dir:
            raise StageError("Training result missing output_dir")
        if not best_weights:
            raise StageError("Training result missing best_weights")
        if not Path(best_weights).exists():
            raise StageError(f"Best weights file not found: {best_weights}")

        result = {
            "framework": str(framework),
            "model": str(resolved_config.get("model", "")),
            "task": str(resolved_config.get("task", "")),
            "train_output_dir": train_output_dir,
            "best_weights": best_weights,
            "resolved_config": str(resolved_config_path),
        }
        _write_json(output_dir / "train_result.json", result)
        if context:
            context.metadata["model_train"] = result
        return output_dir
```

- [ ] **Step 6: Run model train tests**

Run:

```bash
pytest tests/test_training_stages.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit model train stage**

Run:

```bash
git add pipeline/stages/training.py tests/test_training_stages.py
git commit -m "Add model train stage"
```

---

## Task 6: Persist Stage Metadata Through Manifest

**Files:**
- Modify: `pipeline/manifest.py`
- Modify: `pipeline/pipeline.py`
- Create: `tests/test_annotation_to_unitrain_orchestration.py`

- [ ] **Step 1: Write failing manifest metadata orchestration test**

Create `tests/test_annotation_to_unitrain_orchestration.py`:

```python
from pathlib import Path

from pipeline.config import InputConfig, PipelineConfig, Sam2Config
from pipeline.manifest import Manifest
from pipeline.pipeline import PipelineOrchestrator


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
        return output_dir


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
    assert manifest.stages["model_train"]["metadata"]["best_weights"].endswith("checkpoint_best_ema.pth")
    assert manifest.metadata["model_train"]["best_weights"].endswith("checkpoint_best_ema.pth")
```

- [ ] **Step 2: Run metadata test and verify failure**

Run:

```bash
pytest tests/test_annotation_to_unitrain_orchestration.py -v
```

Expected: FAIL because `mark_stage_done` does not accept metadata and `StageContext.metadata` is not connected to manifest metadata.

- [ ] **Step 3: Extend manifest stage metadata**

Change `Manifest.mark_stage_done` in `pipeline/manifest.py` to:

```python
    def mark_stage_done(
        self,
        name: str,
        output_dir: str,
        duration_s: float,
        metadata: dict | None = None,
    ):
        self.stages[name] = {
            "status": "done",
            "output_dir": output_dir,
            "duration_s": duration_s,
        }
        if metadata:
            self.stages[name]["metadata"] = metadata
```

- [ ] **Step 4: Pass manifest metadata into stage context**

In `PipelineOrchestrator._build_stage_context`, change the `RunContext` metadata argument to:

```python
            metadata=manifest.metadata,
```

- [ ] **Step 5: Save stage metadata after successful stages**

In both `run_preset` and `run_stage`, replace:

```python
                manifest.mark_stage_done(stage_name, str(result_path), elapsed)
```

with:

```python
                stage_metadata = {}
                if stage_context.metadata.get(stage_name):
                    stage_metadata = stage_context.metadata[stage_name]
                manifest.mark_stage_done(stage_name, str(result_path), elapsed, metadata=stage_metadata)
```

- [ ] **Step 6: Run metadata orchestration test**

Run:

```bash
pytest tests/test_annotation_to_unitrain_orchestration.py -v
```

Expected: PASS.

- [ ] **Step 7: Run focused pipeline tests**

Run:

```bash
pytest tests/test_standalone_layout.py tests/test_annotation_to_unitrain_orchestration.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit manifest metadata**

Run:

```bash
git add pipeline/manifest.py pipeline/pipeline.py tests/test_annotation_to_unitrain_orchestration.py
git commit -m "Persist training metadata in pipeline manifest"
```

---

## Task 7: Verify End-to-End Configuration and Backward Compatibility

**Files:**
- Modify: `README.md`
- Modify: `tests/test_standalone_layout.py`

- [ ] **Step 1: Add documentation for the new pipeline**

Add this section to `README.md` after the UniTrain COCO output section:

````markdown
## End-to-End UniTrain Pipeline

Use `pipeline: annotation_to_unitrain` when a task should generate an annotation dataset and then train a model with UniTrain.

```yaml
task_id: mouse_001
pipeline: annotation_to_unitrain
runtime: server
class_id: 0
input:
  rgbd_dir: ./tasks/mouse_001/
  video_path: ./tasks/mouse_001/source.mp4
  frame_interval: 1
sam2:
  points: [[380, 182]]
  labels: [1]
detection_dataset:
  class_name: object
  class_id: 0
  train_ratio: 0.8
training: rfdetr_seg_nano
training_overrides:
  train:
    epochs: 20
    batch: 4
    device: 0
output_dir: output/
```

Run it through the same task-config-driven CLI:

```bash
python -m pipeline.cli run --config tasks/mouse_001/task.yaml --force
```

The final weights are recorded in:

```text
output/<task>/model_train/train_result.json
output/<task>/manifest.json
```
````

- [ ] **Step 2: Add test for resolved config writing training data**

Append to `tests/test_standalone_layout.py`:

```python
def test_resolved_config_includes_training_block(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: mouse_001
pipeline: annotation_to_unitrain
runtime: server
input:
  rgbd_dir: ./tasks/mouse_001/
sam2:
  points: [[10, 20]]
  labels: [1]
training: rfdetr_seg_nano
training_overrides:
  train:
    epochs: 3
output_dir: output/
""",
        encoding="utf-8",
    )
    config = load_config(str(task_dir / "task.yaml"), project_root=ROOT)

    from pipeline.pipeline import PipelineOrchestrator

    orch = PipelineOrchestrator(project_root=ROOT)
    path = orch._write_resolved_config(config)
    text = Path(path).read_text(encoding="utf-8")

    assert "training_name: rfdetr_seg_nano" in text
    assert "framework: rfdetr" in text
    assert "epochs: 3" in text
```

- [ ] **Step 3: Run backward compatibility tests**

Run:

```bash
pytest tests/test_standalone_layout.py tests/test_coco_export_and_video_input.py tests/test_sam2_video_cli.py -v
```

Expected: PASS.

- [ ] **Step 4: Run all automated tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 5: Commit docs and compatibility checks**

Run:

```bash
git add README.md tests/test_standalone_layout.py
git commit -m "Document annotation to UniTrain pipeline"
```

---

## Task 8: Final Review and Manual Verification Notes

**Files:**
- No product file changes unless previous tasks expose a defect.

- [ ] **Step 1: Inspect git status**

Run:

```bash
git status --short
```

Expected: no unstaged or staged files after the previous commits.

- [ ] **Step 2: Inspect commit series**

Run:

```bash
git log --oneline -8
```

Expected: the latest commits include:

```text
Document annotation to UniTrain pipeline
Persist training metadata in pipeline manifest
Add model train stage
Add dataset prepare stage
Add dataset manifest validation
Add training preset config layer
Migrate UniTrain into annotation pipeline root
Add annotation to UniTrain design spec
```

- [ ] **Step 3: Record manual training verification command**

Use this command on a machine with SAM2, framework vendors, virtual environments, and GPU access prepared:

```bash
python -m pipeline.cli run --config tasks/mouse_001/task.yaml --force
```

Expected end state:

```text
output/mouse_001/dataset_prepare/dataset_manifest.json
output/mouse_001/model_train/resolved_unitrain_config.yaml
output/mouse_001/model_train/train_result.json
```

`train_result.json` must contain a `best_weights` path that exists on disk.

- [ ] **Step 4: Commit verification notes if README changed during review**

If Step 3 adds a short verification note to `README.md`, run:

```bash
git add README.md
git commit -m "Add manual UniTrain verification note"
```

If `README.md` is unchanged, skip this commit.

---

## Self-Review Checklist

- Spec coverage:
  - Unified root: Task 1.
  - Training config layer: Task 2.
  - Dataset middle layer: Tasks 3 and 4.
  - Training stage and weights result: Task 5.
  - Manifest metadata: Task 6.
  - CLI and backward compatibility: Task 7.
  - Verification handoff: Task 8.
- Red-flag scan: no blocked planning markers are intentionally present in this plan.
- Type consistency:
  - `PipelineConfig.training_name` is `str | None`.
  - `PipelineConfig.training` is `dict[str, Any]`.
  - Dataset manifest path is always `dataset_manifest.json`.
  - Training result path is always `train_result.json`.
  - Manifest training metadata is keyed by `"model_train"`.
