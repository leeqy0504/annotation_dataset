from pathlib import Path
from types import SimpleNamespace

from tools.sam2 import sam2_video_cli


def test_prepare_predictor_frames_converts_png_frames_to_jpg_workspace(tmp_path):
    video_dir = tmp_path / "rgb"
    output_dir = tmp_path / "masks"
    video_dir.mkdir()
    output_dir.mkdir()
    (video_dir / "00000.png").write_bytes(b"png-0")
    (video_dir / "00001.png").write_bytes(b"png-1")

    writes = []
    fake_cv2 = SimpleNamespace(
        imread=lambda path: f"image:{Path(path).name}",
        imwrite=lambda path, image: writes.append((Path(path), image)) or True,
    )

    prepared_dir, output_names = sam2_video_cli._prepare_predictor_frames(
        video_dir,
        output_dir,
        fake_cv2,
    )

    assert prepared_dir == output_dir / "_sam2_jpg_frames"
    assert output_names == ["00000.png", "00001.png"]
    assert writes == [
        (prepared_dir / "00000.jpg", "image:00000.png"),
        (prepared_dir / "00001.jpg", "image:00001.png"),
    ]


def test_prepare_predictor_frames_uses_existing_jpg_frames(tmp_path):
    video_dir = tmp_path / "rgb"
    output_dir = tmp_path / "masks"
    video_dir.mkdir()
    output_dir.mkdir()
    (video_dir / "00000.jpg").write_bytes(b"jpg-0")

    fake_cv2 = SimpleNamespace()

    prepared_dir, output_names = sam2_video_cli._prepare_predictor_frames(
        video_dir,
        output_dir,
        fake_cv2,
    )

    assert prepared_dir == video_dir
    assert output_names == ["00000.jpg"]
