import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    assert config.sam2.project_mount == "/home/try/code/annotation_dataset"
    assert config.detection_dataset.class_name == "object"


def test_only_annotation_dataset_runtime_stages_are_registered():
    assert set(list_stages()) == {
        "masks",
        "prompt_mask",
        "sam2_video_propagation",
        "mask_qa",
        "review_pack",
        "detection_dataset_export",
    }


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
    assert "rgbd_dir: ./tasks/mouse_001/" in task_yaml
    assert "multi_views_dir" not in task_yaml
    assert "real_size" not in task_yaml
