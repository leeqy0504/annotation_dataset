import json
from pathlib import Path
from unittest.mock import patch

from pipeline.config import PipelineConfig, InputConfig, Sam2Config
from pipeline.stages.annotation_dataset import DetectionDatasetExportStage, _write_json
from pipeline.stages.context import DataContext, RunContext, StageContext
from pipeline.stages.sam2_video import ensure_rgb_frames


def _minimal_config(task_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        task="mouse_001",
        preset="annotation_dataset",
        input=InputConfig(
            rgbd_dir=str(task_dir),
        ),
        sam2=Sam2Config(container="sam2-backend-1", points=[[1, 1]], labels=[1]),
    )


def _write_png(path: Path, width: int = 4, height: int = 3) -> None:
    import struct
    import zlib

    raw_rows = []
    for _ in range(height):
        raw_rows.append(b"\x00" + (b"\xff\x00\x00" * width))
    payload = zlib.compress(b"".join(raw_rows))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", payload)
        + chunk(b"IEND", b"")
    )


def _context_for(mask_qa_dir: Path, output_dir: Path) -> StageContext:
    return StageContext(
        run=RunContext(run_id=None, task_name="mouse_001"),
        data=DataContext(
            task_dir=mask_qa_dir.parent,
            run_dir=output_dir.parent,
            output_dir=output_dir,
            inputs={"mask_qa": mask_qa_dir},
        ),
        stage_name="detection_dataset_export",
    )


def test_detection_dataset_export_writes_coco_bbox_and_segmentation(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    rgb_dir = task_dir / "rgb"
    masks_dir = tmp_path / "sam2" / "masks"
    mask_qa_dir = tmp_path / "mask_qa"
    output_dir = tmp_path / "export"
    frame_name = "000000.png"
    _write_png(rgb_dir / frame_name)
    _write_png(masks_dir / frame_name)
    mask_qa_dir.mkdir(parents=True)
    _write_json(mask_qa_dir / "qa_report.json", {
        "task": "mouse_001",
        "source_masks": str(masks_dir),
        "frames": [{
            "frame": frame_name,
            "width": 4,
            "height": 3,
            "area": 4,
            "bbox_xyxy": [1, 0, 3, 2],
            "state": "accepted",
            "flags": [],
        }],
    })

    DetectionDatasetExportStage().run(
        _minimal_config(task_dir),
        output_dir,
        context=_context_for(mask_qa_dir, output_dir),
    )

    coco = json.loads((output_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert coco["info"]["description"] == "mouse_001 train annotation dataset"
    assert coco["images"] == [{
        "id": 1,
        "file_name": "000000.png",
        "width": 4,
        "height": 3,
    }]
    assert coco["categories"] == [{"id": 0, "name": "object", "supercategory": "object"}]
    assert coco["annotations"][0]["bbox"] == [1, 0, 2, 2]
    assert coco["annotations"][0]["area"] == 4
    assert coco["annotations"][0]["segmentation"] == {"size": [3, 4], "counts": [0, 12]}
    assert coco["annotations"][0]["iscrowd"] == 0
    assert (output_dir / "train" / "000000.png").exists()
    assert not (output_dir / "annotations.json").exists()
    assert not (output_dir / "images").exists()
    assert not (output_dir / "labels").exists()
    assert not (output_dir / "dataset.yaml").exists()


def test_detection_dataset_export_splits_by_contiguous_clips(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    rgb_dir = task_dir / "rgb"
    masks_dir = tmp_path / "sam2" / "masks"
    mask_qa_dir = tmp_path / "mask_qa"
    output_dir = tmp_path / "export"
    frames = []
    for index in range(6):
        frame_name = f"{index:06d}.png"
        frames.append({
            "frame": frame_name,
            "width": 4,
            "height": 3,
            "area": 4,
            "bbox_xyxy": [1, 0, 3, 2],
            "state": "accepted",
            "flags": [],
        })
        _write_png(rgb_dir / frame_name)
        _write_png(masks_dir / frame_name)
    mask_qa_dir.mkdir(parents=True)
    _write_json(mask_qa_dir / "qa_report.json", {
        "task": "mouse_001",
        "source_masks": str(masks_dir),
        "frames": frames,
    })
    config = _minimal_config(task_dir)
    config.detection_dataset.clip_size = 2
    config.detection_dataset.train_ratio = 0.5

    DetectionDatasetExportStage().run(
        config,
        output_dir,
        context=_context_for(mask_qa_dir, output_dir),
    )

    train = json.loads((output_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    valid = json.loads((output_dir / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert [image["file_name"] for image in train["images"]] == ["000000.png", "000001.png"]
    assert [image["file_name"] for image in valid["images"]] == [
        "000002.png",
        "000003.png",
        "000004.png",
        "000005.png",
    ]
    assert (output_dir / "train" / "000000.png").exists()
    assert (output_dir / "valid" / "000002.png").exists()


def test_detection_dataset_export_adapts_clip_size_for_default_80_20_split(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    rgb_dir = task_dir / "rgb"
    masks_dir = tmp_path / "sam2" / "masks"
    mask_qa_dir = tmp_path / "mask_qa"
    output_dir = tmp_path / "export"
    frames = []
    for index in range(80):
        frame_name = f"{index:06d}.png"
        frames.append({
            "frame": frame_name,
            "width": 4,
            "height": 3,
            "area": 4,
            "bbox_xyxy": [1, 0, 3, 2],
            "state": "accepted",
            "flags": [],
        })
        _write_png(rgb_dir / frame_name)
        _write_png(masks_dir / frame_name)
    mask_qa_dir.mkdir(parents=True)
    _write_json(mask_qa_dir / "qa_report.json", {
        "task": "mouse_001",
        "source_masks": str(masks_dir),
        "frames": frames,
    })

    DetectionDatasetExportStage().run(
        _minimal_config(task_dir),
        output_dir,
        context=_context_for(mask_qa_dir, output_dir),
    )

    train = json.loads((output_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    valid = json.loads((output_dir / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert len(train["images"]) == 64
    assert len(valid["images"]) == 16
    assert train["images"][0]["file_name"] == "000000.png"
    assert train["images"][-1]["file_name"] == "000063.png"
    assert valid["images"][0]["file_name"] == "000064.png"
    assert valid["images"][-1]["file_name"] == "000079.png"
    assert train["metadata"]["clip_size"] == 8
    assert train["metadata"]["clip_size_mode"] == "adaptive"


def test_video_input_extracts_rgb_frames_with_ffmpeg(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse_001"
    video_path = task_dir / "source.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    config = _minimal_config(task_dir)
    config.input.video_path = str(video_path)
    config.input.frame_interval = 3

    def fake_run(cmd, capture_output, text):
        assert cmd[:4] == ["ffmpeg", "-y", "-i", str(video_path)]
        assert "select=not(mod(n\\,3))" in cmd
        assert str(task_dir / "rgb" / "%06d.png") in cmd
        _write_png(task_dir / "rgb" / "000000.png")
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    with patch("pipeline.stages.sam2_video.subprocess.run", side_effect=fake_run) as run:
        rgb_dir = ensure_rgb_frames(config)

    assert rgb_dir == task_dir / "rgb"
    assert (rgb_dir / "000000.png").exists()
    assert run.call_count == 1
