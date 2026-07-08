"""Pipeline orchestrator: resolve preset, run stages in sequence."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pipeline.config import PipelineConfig
from pipeline.manifest import Manifest
from pipeline.stages import get_stage

if TYPE_CHECKING:
    from pipeline.stages.context import StageContext


class PipelineOrchestrator:
    """Runs presets by dispatching stages in order, tracking progress via manifest."""

    def __init__(self, presets_path: str | None = None,
                 project_root: str | Path | None = None):
        if presets_path is not None:
            self.presets = self._load_presets_file(Path(presets_path))
            return

        if project_root is not None:
            pipeline_dir = Path(project_root) / "configs" / "pipelines"
            if pipeline_dir.exists():
                presets = self._load_pipeline_dir(pipeline_dir)
                if presets:
                    self.presets = presets
                    return

        self.presets = self._load_presets_file(Path(__file__).parent / "presets.yaml")

    @staticmethod
    def _load_presets_file(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_pipeline_dir(path: Path) -> dict:
        presets = {}
        for config_path in sorted([*path.glob("*.yaml"), *path.glob("*.yml")]):
            data = PipelineOrchestrator._load_presets_file(config_path)
            preset = data.get("preset") or config_path.stem
            presets[preset] = {
                "stages": list(data.get("stages", [])),
            }
        return presets

    def resolve_preset(self, preset_name: str) -> list[str]:
        if preset_name not in self.presets:
            raise KeyError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(self.presets.keys())}"
            )
        return list(self.presets[preset_name]["stages"])

    def resolve_stages(self, config: PipelineConfig) -> list[str]:
        if config.pipeline_stages:
            return list(config.pipeline_stages)
        return self.resolve_preset(config.preset)

    def _run_dir(self, config: PipelineConfig) -> str:
        task_dir = Path(config.output_dir) / config.task
        if config.run_id:
            return str(task_dir / "runs" / config.run_id)
        return str(task_dir)

    def _manifest_path(self, config: PipelineConfig) -> str:
        return str(Path(self._run_dir(config)) / "manifest.json")

    def _stage_output_dir(self, config: PipelineConfig, stage_name: str) -> str:
        base = Path(self._run_dir(config))
        if config.run_id:
            return str(base / "stages" / stage_name)
        return str(base / stage_name)

    def _write_resolved_config(self, config: PipelineConfig) -> str:
        path = Path(self._run_dir(config)) / "resolved_config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task": config.task,
            "preset": config.preset,
            "run_id": config.run_id,
            "input": config.input.__dict__,
            "sam2": config.sam2.__dict__,
            "detection_dataset": config.detection_dataset.__dict__,
            "training_name": config.training_name,
            "training": config.training,
            "output_dir": config.output_dir,
            "pipeline_stages": config.pipeline_stages,
            "runtime": config.runtime,
            "registry_snapshot": config.registry_snapshot,
            "source_config_path": config.source_config_path,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return str(path)

    def _build_stage_context(
        self,
        config: PipelineConfig,
        stage_name: str,
        output_dir: Path,
        manifest: Manifest,
        base_context: "StageContext | None" = None,
        resolved_config_path: str | None = None,
    ) -> "StageContext":
        from pipeline.stages.context import DataContext, RunContext, StageContext

        inputs = {
            name: Path(info["output_dir"])
            for name, info in manifest.stages.items()
            if info.get("status") == "done" and info.get("output_dir")
        }
        run_dir = Path(self._run_dir(config))
        task_dir = Path(config.input.rgbd_dir)
        metadata = manifest.metadata
        run_context = RunContext(
            run_id=config.run_id,
            task_name=config.task,
            logger=base_context.logger if base_context else None,
            resolved_config_path=Path(resolved_config_path) if resolved_config_path else None,
            job_id=(base_context.job_id if base_context else None) or config.run_id,
            stop_event=base_context.stop_event if base_context else None,
            progress_callback=base_context.progress_callback if base_context else None,
            metadata=metadata,
        )
        data_context = DataContext(
            task_dir=task_dir,
            run_dir=run_dir,
            output_dir=output_dir,
            inputs=inputs,
        )
        return StageContext(run=run_context, data=data_context, stage_name=stage_name)

    def run_preset(self, config: PipelineConfig, force: bool = False,
                   context: "StageContext | None" = None,
                   enabled_stages: list[str] | None = None):
        stages = self.resolve_stages(config)
        enabled = set(enabled_stages) if enabled_stages is not None else None

        manifest_path = self._manifest_path(config)
        resolved_config_path = self._write_resolved_config(config)
        manifest = Manifest.load(manifest_path) if Path(manifest_path).exists() else Manifest(
            task=config.task,
            config_path=resolved_config_path,
            run_id=config.run_id,
        )
        manifest.config_path = resolved_config_path
        if enabled is not None:
            skipped = [name for name in stages if name not in enabled]
            manifest.metadata["stage_selection"] = {
                "preset": config.preset,
                "enabled": [name for name in stages if name in enabled],
                "skipped": skipped,
            }
        manifest.metadata["run_dir"] = self._run_dir(config)
        if config.registry_snapshot:
            manifest.metadata["registry_snapshot"] = config.registry_snapshot

        for stage_name in stages:
            if enabled is not None and stage_name not in enabled:
                manifest.mark_stage_skipped(stage_name)
                manifest.save(manifest_path)
                print(f"[pipeline] {stage_name}: skipped")
                continue

            if not force and manifest.is_stage_done(stage_name):
                print(f"[pipeline] {stage_name}: skip (already done)")
                continue

            print(f"[pipeline] {stage_name}: starting...")
            stage = get_stage(stage_name)
            output_dir = Path(self._stage_output_dir(config, stage_name))
            stage_context = self._build_stage_context(
                config, stage_name, output_dir, manifest, context, resolved_config_path)

            start = time.time()
            try:
                result_path = stage.run(config, output_dir, context=stage_context)
                elapsed = time.time() - start
                stage_metadata = {}
                if stage_context.metadata.get(stage_name):
                    stage_metadata = stage_context.metadata[stage_name]
                manifest.mark_stage_done(
                    stage_name,
                    str(result_path),
                    elapsed,
                    metadata=stage_metadata,
                )
                print(f"[pipeline] {stage_name}: done ({elapsed:.1f}s)")
            except Exception as e:
                manifest.mark_stage_failed(stage_name)
                manifest.save(manifest_path)
                print(f"[pipeline] {stage_name}: FAILED - {e}")
                raise

            manifest.save(manifest_path)

        print(f"[pipeline] Complete. Manifest: {manifest_path}")

    def run_stage(self, config: PipelineConfig, stage_name: str, force: bool = False,
                  context: "StageContext | None" = None):
        manifest_path = self._manifest_path(config)
        resolved_config_path = self._write_resolved_config(config)
        manifest = Manifest.load(manifest_path) if Path(manifest_path).exists() else Manifest(
            task=config.task,
            config_path=resolved_config_path,
            run_id=config.run_id,
        )
        manifest.config_path = resolved_config_path
        manifest.metadata["run_dir"] = self._run_dir(config)

        if not force and manifest.is_stage_done(stage_name):
            print(f"[pipeline] {stage_name}: skip (already done)")
            return

        print(f"[pipeline] {stage_name}: starting...")
        stage = get_stage(stage_name)
        output_dir = Path(self._stage_output_dir(config, stage_name))
        stage_context = self._build_stage_context(
            config, stage_name, output_dir, manifest, context, resolved_config_path)

        start = time.time()
        try:
            result_path = stage.run(config, output_dir, context=stage_context)
            elapsed = time.time() - start
            stage_metadata = {}
            if stage_context.metadata.get(stage_name):
                stage_metadata = stage_context.metadata[stage_name]
            manifest.mark_stage_done(
                stage_name,
                str(result_path),
                elapsed,
                metadata=stage_metadata,
            )
            print(f"[pipeline] {stage_name}: done ({elapsed:.1f}s)")
        except Exception as e:
            manifest.mark_stage_failed(stage_name)
            manifest.save(manifest_path)
            print(f"[pipeline] {stage_name}: FAILED - {e}")
            raise

        manifest.save(manifest_path)

    def status(self, config: PipelineConfig):
        manifest_path = self._manifest_path(config)
        if not Path(manifest_path).exists():
            print(f"No manifest found for task '{config.task}'")
            return

        manifest = Manifest.load(manifest_path)
        print(f"Task: {manifest.task}")
        print(f"Created: {manifest.created_at}")
        print("Stages:")
        for name, info in manifest.stages.items():
            status = info["status"]
            marker = "✅" if status == "done" else "❌" if status == "failed" else "⏳"
            output = info.get("output_dir") or "-"
            duration = info.get("duration_s")
            dur_str = f" ({duration:.1f}s)" if duration else ""
            print(f"  {marker} {name}: {status}{dur_str} -> {output}")
