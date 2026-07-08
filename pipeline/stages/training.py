"""Training-related pipeline stages."""

import copy
import json
from pathlib import Path

import yaml

from datasets import DatasetValidationError, prepare_dataset_manifest
from pipeline.config import PipelineConfig
from pipeline.manifest import load_manifest_for_config
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext
from unitrain import get_runner


def _stage_input(config: PipelineConfig, context: StageContext | None, stage_name: str) -> Path:
    if context and context.data and context.data.get_input(stage_name):
        return context.input(stage_name)
    manifest = load_manifest_for_config(config)
    output = manifest.get_output_dir(stage_name)
    if output:
        return Path(output)
    raise StageError(f"Required stage input not found: {stage_name}")


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
    if not isinstance(data, dict):
        raise StageError("Training config field 'data' must be a mapping")
    data["path"] = dataset_manifest["root"]
    data.setdefault("format", dataset_manifest.get("format", "coco").replace("roboflow_", ""))
    return resolved


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
        try:
            runner = get_runner(str(framework))
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
