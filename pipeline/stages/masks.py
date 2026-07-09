"""SAM2 mask generation stage: runs SAM2 via a running Docker container."""

import json
import logging
import subprocess
from pathlib import Path
from shlex import quote

from pipeline.config import PipelineConfig
from pipeline.input_source import dataset_info_candidates, input_root, rgb_frame_files
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext
from pipeline.stages.sam2_video import _container_path, ensure_rgb_frames


@register_stage("masks")
@register_stage("prompt_mask")
class Sam2MaskStage(BaseStage):
    name = "masks"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        task_dir = input_root(config)
        self.check_input_path(str(task_dir), "input source directory")

        rgb_src = ensure_rgb_frames(config)
        self.check_input_path(str(rgb_src), "RGB source directory")

        rgb_files = rgb_frame_files(config)
        if not rgb_files:
            raise StageError(f"No RGB images found in {rgb_src}")

        first_frame = rgb_files[config.input.first_frame]
        container = config.sam2.container

        # 1. Resolve points/labels — prefer dataset_info.json over config
        points = config.sam2.points
        labels = config.sam2.labels

        dataset_info_path = None
        for p in dataset_info_candidates(config):
            if p.exists():
                dataset_info_path = p
                break

        if dataset_info_path:
            with open(dataset_info_path) as f:
                ds = json.load(f)
            sam2_data = ds.get("sam2_points", {})
            pts = sam2_data.get("points", [])
            lbls = sam2_data.get("labels", [])
            if pts and lbls and len(pts) == len(lbls):
                points = pts
                labels = lbls
                if context:
                    context.log(logging.INFO, "Using %d point(s) from %s", len(points), dataset_info_path)
                else:
                    print(f"[sam2mask] Using {len(points)} point(s) from {dataset_info_path}")
            else:
                if context:
                    context.log(logging.WARNING, "dataset_info.json found but sam2_points invalid, using config values")
                else:
                    print("[sam2mask] WARNING: dataset_info.json found but sam2_points invalid, using config values")
        else:
            if context:
                context.log(logging.INFO, "No dataset_info.json found, using config pts/labels")
            else:
                print("[sam2mask] No dataset_info.json found, using config pts/labels")

        points_str = " ".join(f"{x},{y}" for x, y in points)
        labels_str = " ".join(str(label) for label in labels)
        if not points or not labels or len(points) != len(labels):
            raise StageError("SAM2 points/labels are required for mask generation")

        mask_output = output_dir / first_frame.name
        project_root = Path(__file__).resolve().parents[2]
        script_host = project_root / config.sam2.pic_cli
        self.check_input_path(str(script_host), "SAM2 picture CLI")
        script_container = _container_path(script_host, project_root, config.sam2.project_mount)
        image_container = _container_path(first_frame, project_root, config.sam2.project_mount)
        output_container = _container_path(mask_output, project_root, config.sam2.project_mount)

        # 2. Run inference inside container using the mounted project directory.
        cmd = (
            f"PYTHONPATH=/opt/sam2/server python {quote(script_container)} "
            f"--image {quote(image_container)} "
            f"--points {quote(points_str)} "
            f"--labels {quote(labels_str)} "
            f"--output {quote(output_container)} "
            f"--checkpoint {quote(config.sam2.checkpoint)} "
            f"--config {quote(config.sam2.config_file)}"
        )
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise StageError(
                f"SAM2 CLI failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        # 3. Parse JSON output from CLI
        payload = json.loads(result.stdout.strip().split("\n")[-1])

        if not mask_output.exists():
            raise StageError(f"SAM2 completed but mask not found at {mask_output}")

        metadata = {
            "frame": first_frame.name,
            "image": str(first_frame),
            "mask": str(mask_output),
            "points": points,
            "labels": labels,
            "score": payload.get("score"),
            "foreground_pixels": payload.get("foreground_pixels"),
            "mask_shape": payload.get("mask_shape"),
            "cli": config.sam2.pic_cli,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output_dir
