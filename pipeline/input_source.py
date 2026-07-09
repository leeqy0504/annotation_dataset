"""Input source helpers for RGB frames and video extraction."""

from pathlib import Path

from pipeline.config import PipelineConfig


IMAGE_EXTS = {".png"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def input_root(config: PipelineConfig) -> Path:
    """Return the task/input root used for dataset_info and context metadata."""
    source = getattr(config.input, "source", None)
    if source:
        path = Path(source)
        return path.parent if path.is_file() else path
    return Path(config.input.rgbd_dir)


def rgb_frame_dir(config: PipelineConfig) -> Path:
    """Return the directory that contains the canonical RGB frame sequence."""
    source = getattr(config.input, "source", None)
    if source:
        path = Path(source)
        if path.suffix.lower() in VIDEO_EXTS:
            return path.parent / path.stem
        return path
    return Path(config.input.rgbd_dir) / "rgb"


def rgb_frame_files(config: PipelineConfig) -> list[Path]:
    """Return sorted PNG frames from the configured RGB frame directory."""
    frame_dir = rgb_frame_dir(config)
    if not frame_dir.exists():
        return []
    return sorted(
        path for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def dataset_info_candidates(config: PipelineConfig) -> list[Path]:
    """Return likely dataset_info.json locations for old and new layouts."""
    candidates = [input_root(config) / "dataset_info.json"]
    rgbd_dir = getattr(config.input, "rgbd_dir", None)
    if rgbd_dir:
        legacy = Path(rgbd_dir) / "dataset_info.json"
        if legacy not in candidates:
            candidates.append(legacy)
    return candidates


def discover_video_source(config: PipelineConfig) -> Path | None:
    """Find an explicit or colocated video source for frame extraction."""
    video_path = getattr(config.input, "video_path", None)
    if video_path:
        return Path(video_path)

    source = getattr(config.input, "source", None)
    if not source:
        return None
    path = Path(source)
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return path
    if path.is_dir():
        videos = sorted(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in VIDEO_EXTS
        )
        if videos:
            return videos[0]
    return None


def next_frame_number(frames: list[Path]) -> int:
    """Return the next numeric frame index, preserving existing frames."""
    numeric = [int(path.stem) for path in frames if path.stem.isdigit()]
    if numeric:
        return max(numeric) + 1
    return len(frames)
