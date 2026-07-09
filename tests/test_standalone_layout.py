import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from pipeline.config import load_config
from pipeline.stages import list_stages


ROOT = Path(__file__).resolve().parents[1]


def test_layered_annotation_dataset_config_loads_from_task_yaml(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: mouse_001
pipeline: annotation_dataset
runtime: server
class_id: 0
input:
  rgbd_dir: ./tasks/mouse_001/
  video_path: ./tasks/mouse_001/source.mp4
  frame_interval: 2
sam2:
  points: [[10, 20]]
  labels: [1]
detection_dataset:
  clip_size: 500
  train_ratio: 0.8
output_dir: output/
""",
        encoding="utf-8",
    )

    config = load_config(str(task_dir / "task.yaml"), project_root=ROOT)

    assert config.task == "mouse_001"
    assert config.preset == "annotation_dataset"
    assert config.pipeline_stages == [
        "prompt_mask",
        "sam2_video_propagation",
        "mask_qa",
        "review_pack",
        "detection_dataset_export",
    ]
    assert config.input.video_path == "./tasks/mouse_001/source.mp4"
    assert config.input.frame_interval == 2
    assert config.sam2.container == "sam2-backend-1"
    assert config.sam2.project_mount == "/home/try/code/perception-platform"
    assert config.detection_dataset.class_name == "object"
    assert config.detection_dataset.clip_size == 500
    assert config.detection_dataset.train_ratio == 0.8


def test_layered_config_accepts_source_without_rgbd_dir(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: mouse_001
pipeline: annotation_dataset
runtime: server
class_id: 0
input:
  source: ./tasks/mouse_001/
  first_frame: 0
sam2:
  points: [[10, 20]]
  labels: [1]
output_dir: output/
""",
        encoding="utf-8",
    )

    config = load_config(str(task_dir / "task.yaml"), project_root=ROOT)

    assert config.input.source == "./tasks/mouse_001/"
    assert config.input.rgbd_dir == "./tasks/mouse_001/"


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


def test_legacy_preset_resolver_includes_annotation_to_unitrain():
    from pipeline.pipeline import PipelineOrchestrator

    stages = PipelineOrchestrator().resolve_preset("annotation_to_unitrain")

    assert stages == [
        "prompt_mask",
        "sam2_video_propagation",
        "mask_qa",
        "review_pack",
        "detection_dataset_export",
        "dataset_prepare",
        "model_train",
    ]
    assert stages[-2:] == ["dataset_prepare", "model_train"]


def test_runner_invokes_pipeline_from_project_root(tmp_path):
    task_dir = ROOT / "tasks" / "__runner_test__"
    config_path = task_dir / "task.yaml"
    task_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
task_id: __runner_test__
pipeline: annotation_dataset
runtime: server
input:
  rgbd_dir: ./tasks/__runner_test__/
sam2:
  points: [[1, 2]]
  labels: [1]
output_dir: output/
""",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "capture.json"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
Path({str(capture_path)!r}).write_text(json.dumps({{
    "argv": sys.argv,
    "cwd": os.getcwd(),
}}))
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    try:
        env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
        subprocess.run(
            [str(ROOT / "run_annotation_dataset.sh"), "--task", "__runner_test__", "--force"],
            cwd="/",
            env=env,
            check=True,
        )
        captured = json.loads(capture_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert captured["cwd"] == str(ROOT)
    assert captured["argv"][1:] == [
        "-m",
        "pipeline.cli",
        "run",
        "--config",
        str(config_path),
        "--force",
    ]


def test_setup_command_writes_annotation_dataset_task_without_multiviews(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    task_dir.mkdir(parents=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.cli",
            "setup",
            "--task",
            "mouse_001",
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )

    task_yaml = (task_dir / "task.yaml").read_text(encoding="utf-8")
    assert "pipeline: annotation_dataset" in task_yaml
    assert "source: ./tasks/mouse_001/" in task_yaml
    assert "rgbd_dir" not in task_yaml
    assert "multi_views_dir" not in task_yaml
    assert "real_size" not in task_yaml


def test_project_identity_uses_perception_platform_name():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "perception-platform"' in pyproject
    assert 'perception-platform = "pipeline.cli:main"' in pyproject


def test_example_yaml_files_cover_common_workflows():
    expected = {
        "dataset_only.yaml",
        "train_yolo.yaml",
        "end_to_end_yolo.yaml",
        "end_to_end_rfdetr.yaml",
        "mixed_images_and_video.yaml",
    }
    example_dir = ROOT / "examples"

    assert expected <= {path.name for path in example_dir.glob("*.yaml")}

    dataset_only = yaml.safe_load((example_dir / "dataset_only.yaml").read_text(encoding="utf-8"))
    train_yolo = yaml.safe_load((example_dir / "train_yolo.yaml").read_text(encoding="utf-8"))
    end_to_end_yolo = yaml.safe_load((example_dir / "end_to_end_yolo.yaml").read_text(encoding="utf-8"))
    mixed = yaml.safe_load((example_dir / "mixed_images_and_video.yaml").read_text(encoding="utf-8"))

    assert dataset_only["pipeline"] == "annotation_dataset"
    assert dataset_only["input"]["source"] == "./tasks/mouse_001/"
    assert train_yolo["framework"] == "ultralytics"
    assert train_yolo["data"]["format"] == "coco"
    assert end_to_end_yolo["pipeline"] == "annotation_to_unitrain"
    assert end_to_end_yolo["training"] == "yolo11n_seg"
    assert mixed["input"]["source"] == "./tasks/videotest/"
    assert mixed["input"]["video_path"] == "./tasks/videotest/source.mp4"


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


def test_resolved_config_includes_training_block(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    output_dir = tmp_path / "output"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        f"""
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
output_dir: {output_dir}
""",
        encoding="utf-8",
    )
    config = load_config(str(task_dir / "task.yaml"), project_root=ROOT)

    from pipeline.pipeline import PipelineOrchestrator

    orch = PipelineOrchestrator(project_root=ROOT)
    path = orch._write_resolved_config(config)
    resolved = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    assert Path(path) == output_dir / "mouse_001" / "resolved_config.yaml"
    assert not (ROOT / "output" / "mouse_001" / "resolved_config.yaml").exists()
    assert resolved["training_name"] == "rfdetr_seg_nano"
    assert resolved["training"]["framework"] == "rfdetr"
    assert resolved["training"]["train"]["epochs"] == 3
