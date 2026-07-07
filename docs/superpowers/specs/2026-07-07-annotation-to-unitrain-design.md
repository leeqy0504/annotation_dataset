# Annotation to UniTrain End-to-End Design

Date: 2026-07-07
Status: Approved for implementation planning

## Context

The current workspace contains two separate projects:

- `annotation-dataset-pipeline`: a staged CLI pipeline that generates SAM2 masks and exports a Roboflow-style COCO dataset.
- `unitrain-dev`: a unified training framework that can train RF-DETR and Ultralytics/YOLO models from COCO data.

The target is an internal perception platform pipeline. Given a video or RGB image folder, the system should generate a dataset and train a model, ending with a model weight file path that can be read from pipeline outputs.

The architecture must not keep the two systems as separate project folders. `annotation-dataset-pipeline` will become the unified project root, and UniTrain will be migrated into it as downstream training capability.

## Goals

- Build an MVP end-to-end CLI path:
  input video or RGB folder -> annotation dataset -> dataset manifest -> UniTrain training -> model weights.
- Preserve the existing annotation pipeline configuration architecture.
- Add a dataset management middle layer between dataset export and training.
- Support both `rfdetr` and `ultralytics`/`yolo` through the existing UniTrain runner abstraction.
- Keep training framework dependencies isolated by framework.
- Use task-config-driven CLI execution for the MVP.
- Use subagents as a development workflow, not as product runtime architecture.

## Non-Goals

- No frontend implementation in the MVP.
- No product-runtime agent orchestration.
- No full dataset registry or model registry beyond stage-local manifests.
- No unified virtual environment for all training frameworks.
- No large package rename to a new top-level `perception_platform` package in the MVP.

## Chosen Approach

Use stage-native integration.

`annotation-dataset-pipeline` remains the orchestrator. UniTrain is migrated into the same project root as normal training capability, and training becomes ordinary pipeline stages rather than an external script glue layer.

Rejected alternatives:

- Adapter-only integration: faster, but preserves the two-project boundary and weakens platform architecture.
- Full platform package rewrite: cleaner long term, but too much import churn and risk for the first end-to-end milestone.

## Project Layout

The unified project root will be:

```text
annotation-dataset-pipeline/
  pipeline/              # Config loading, orchestrator, stage registry, manifest, CLI
  datasets/              # Dataset manifest, validation, training input adapter
  unitrain/              # Migrated training framework core
  cli/                   # Migrated UniTrain train/eval/export/predict entry modules
  envs/                  # Framework-specific dependency files
  vendors/               # Framework source/vendor checkouts
  configs/
    pipelines/
    algorithms/
    runtime/
    training/            # New training presets
```

The existing `pipeline/` package remains responsible for orchestration. The new `datasets/` package owns the boundary between exported datasets and training. The `unitrain/` package owns training framework selection, data conversion, runners, and evaluation helpers.

## Pipeline Stages

The existing annotation-only pipeline remains supported:

```text
prompt_mask
-> sam2_video_propagation
-> mask_qa
-> review_pack
-> detection_dataset_export
```

The new end-to-end pipeline preset is `annotation_to_unitrain`:

```text
prompt_mask
-> sam2_video_propagation
-> mask_qa
-> review_pack
-> detection_dataset_export
-> dataset_prepare
-> model_train
```

`dataset_prepare` reads the `detection_dataset_export` output and writes a `dataset_manifest.json`. It validates the dataset and records the training input contract without copying image data.

`model_train` reads `dataset_manifest.json`, writes `resolved_unitrain_config.yaml`, calls the UniTrain runner abstraction, and writes `train_result.json`.

## Configuration

The existing layered configuration model is preserved and extended with a training layer.

Example task config:

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

Example training preset:

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

Config merge order:

```text
algorithm defaults
-> runtime config
-> pipeline preset metadata
-> training preset
-> training_overrides
-> task-local non-training fields
```

The task config selects the pipeline, runtime, and training preset. Task-local overrides for training live under `training_overrides`, so task authors can change values such as `train.epochs`, `train.batch`, and `train.device` without copying a full UniTrain config.

`PipelineConfig` will gain a training config block, but UniTrain fields will remain grouped rather than flattened into annotation pipeline fields. `model_train` injects the dataset path at runtime, so task authors do not manually reference previous stage output paths.

## Dataset Contract

The annotation exporter currently emits Roboflow-style COCO:

```text
detection_dataset_export/
  train/
    *.png
    _annotations.coco.json
  valid/
    *.png
    _annotations.coco.json
  masks/
  preview/
```

`dataset_prepare` validates this structure and writes:

```text
dataset_prepare/
  dataset_manifest.json
```

Manifest shape:

```json
{
  "dataset_id": "mouse_001:<run_id>:detection_dataset_export",
  "format": "roboflow_coco",
  "root": ".../detection_dataset_export",
  "splits": {
    "train": {
      "images_dir": ".../train",
      "annotations": ".../train/_annotations.coco.json",
      "image_count": 64,
      "annotation_count": 64
    },
    "valid": {
      "images_dir": ".../valid",
      "annotations": ".../valid/_annotations.coco.json",
      "image_count": 16,
      "annotation_count": 16
    }
  },
  "categories": [{"id": 0, "name": "object"}],
  "source_stage": "detection_dataset_export",
  "validation": {
    "status": "passed",
    "warnings": []
  }
}
```

The MVP uses manifest plus original dataset references. It does not copy the dataset to a separate registry path.

## Training Contract

`model_train` writes:

```text
model_train/
  resolved_unitrain_config.yaml
  train_result.json
```

`resolved_unitrain_config.yaml` contains a normal UniTrain config with `data.path` injected from the dataset manifest root.

`train_result.json` shape:

```json
{
  "framework": "rfdetr",
  "model": "seg-nano",
  "task": "segment",
  "train_output_dir": ".../outputs/rfdetr_...",
  "best_weights": ".../checkpoint_best_ema.pth",
  "resolved_config": ".../model_train/resolved_unitrain_config.yaml"
}
```

The pipeline manifest continues to track stage status. The `model_train` output directory and result file become the stable source for downstream CLI status and future frontend reads.

The training result summary is also mirrored into manifest metadata. The preferred shape is stage-local metadata on the `model_train` stage entry; if the manifest implementation keeps stage entries minimal, the summary must be placed under a namespaced run-level key such as `metadata["model_train"]`. In both cases, `train_result.json` remains the canonical detailed result.

## CLI

The MVP remains task-config driven:

```bash
python -m pipeline.cli run --config tasks/mouse_001/task.yaml --force
```

The existing helper remains valid:

```bash
./run_annotation_dataset.sh --task mouse_001 --force
```

If `task.yaml` uses `pipeline: annotation_dataset`, only the annotation dataset stages run. If it uses `pipeline: annotation_to_unitrain`, the full end-to-end stage chain runs.

Future frontend work can create and edit task configs, then trigger the same CLI/orchestrator path. No input-source-centered CLI is needed for the MVP.

## Environment Management

Keep UniTrain's framework isolation after migration:

- RF-DETR uses `.venv-rfdetr`.
- Ultralytics/YOLO uses `.venv-yolo`.
- Dependency files remain under `envs/`.
- Vendor checkouts remain under `vendors/`.

The training runner remains responsible for verifying and using the correct framework environment. The MVP does not merge training frameworks into one virtual environment.

## Error Handling

Configuration errors fail early with clear messages:

- Missing training preset.
- Invalid `training_overrides` type.
- Missing required SAM2 points or labels.
- Unknown pipeline preset or stage.

Dataset errors fail in `dataset_prepare`:

- Missing split annotation file.
- Missing images referenced by annotations.
- Empty required split.
- Inconsistent categories between splits.
- Unsupported dataset layout.

Training errors fail in `model_train`:

- Runner exits non-zero.
- Framework environment is missing or invalid.
- Training returns no best weight path.
- Best weight path is reported but does not exist.

Failed stages update `manifest.json` through the existing pipeline failure path. `model_train` keeps `resolved_unitrain_config.yaml` when possible to preserve reproducibility.

## Testing

Required tests:

- Config loading: `training: rfdetr_seg_nano` resolves `configs/training/rfdetr_seg_nano.yaml` and merges `training_overrides`.
- Stage registry: `dataset_prepare` and `model_train` are registered without breaking existing annotation-only stage expectations.
- Dataset manifest: temporary Roboflow-style COCO data produces correct split counts, annotation counts, categories, and validation status.
- Training stage: mocked UniTrain runner writes `resolved_unitrain_config.yaml` and `train_result.json`, with `data.path` injected from the dataset manifest.
- End-to-end orchestration: `annotation_to_unitrain` executes the expected stage order and records outputs in `manifest.json`.
- Backward compatibility: existing `annotation_dataset` tests continue to pass.

Heavy training integration tests are out of MVP automated scope. Real framework execution can be verified manually or in an environment that has GPUs, vendors, and framework virtualenvs prepared.

## Subagent Development Workflow

Subagents are a development process, not a product runtime feature.

Roles:

- `a/specs`: produces the architecture/spec and interface decisions.
- `b/reviewer`: reviews the spec and implementation plan for boundaries, config design, dataset contract, and testability.
- `c/builder`: implements the approved plan.
- `d/tester`: validates tests, CLI behavior, manifest output, and error scenarios.

The product code should stay free of these role concepts. This design spec is the MVP workflow record; if implementation needs a standalone handoff template, add `docs/superpowers/agent-workflow.md` with the same four roles and their expected artifacts.

## Implementation Notes

- Treat the existing modified file in `unitrain-dev/unitrain/runners/_scripts/rfdetr_eval.py` as user work during migration.
- Preserve current annotation-only behavior and CLI compatibility.
- Prefer direct Python runner calls from `model_train` over shelling out to a separate project path.
- Keep dataset management in `datasets/`; avoid letting training stages inspect exporter internals directly.
- Keep the first implementation focused on the MVP closed loop. Registry, export automation, richer reports, and frontend integration can build on the manifest contracts later.
