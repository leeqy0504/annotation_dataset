#!/usr/bin/env python3
"""SAM2 video propagation CLI used inside the SAM2 container."""

import argparse
import json
import shutil
from pathlib import Path


def _load_runtime_dependencies():
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    return cv2, np, torch, build_sam2_video_predictor


def _parse_points(text: str, np):
    return np.array([list(map(int, p.split(","))) for p in text.strip().split()], dtype=np.float32)


def _parse_labels(text: str, np):
    return np.array(list(map(int, text.strip().split())), dtype=np.int32)


def _frame_paths(video_dir: Path) -> list[Path]:
    frames = sorted(video_dir.glob("*.png"))
    if not frames:
        frames = sorted(video_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No image frames found in {video_dir}")
    return frames


def _prepare_predictor_frames(video_dir: Path, output_dir: Path, cv2) -> tuple[Path, list[str]]:
    frames = _frame_paths(video_dir)
    output_names = [frame.name for frame in frames]
    if frames[0].suffix.lower() in (".jpg", ".jpeg"):
        return video_dir, output_names

    jpg_dir = output_dir / "_sam2_jpg_frames"
    if jpg_dir.exists():
        shutil.rmtree(jpg_dir)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        image = cv2.imread(str(frame))
        if image is None:
            raise FileNotFoundError(f"Frame read failed: {frame}")
        jpg_path = jpg_dir / f"{frame.stem}.jpg"
        if not cv2.imwrite(str(jpg_path), image):
            raise RuntimeError(f"Frame conversion failed: {jpg_path}")
    return jpg_dir, output_names


def _should_use_prompt_mask(prompt_mask: str) -> bool:
    return bool(prompt_mask and Path(prompt_mask).exists())


def _add_initial_prompt(
    predictor,
    state,
    first_frame: int,
    points,
    labels,
    prompt_mask: str,
    cv2,
) -> str:
    if _should_use_prompt_mask(prompt_mask) and hasattr(predictor, "add_new_mask"):
        mask = cv2.imread(prompt_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Prompt mask read failed: {prompt_mask}")
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=first_frame,
            obj_id=1,
            mask=mask > 0,
        )
        return "mask"

    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=first_frame,
        obj_id=1,
        points=points,
        labels=labels,
    )
    return "points"


def main():
    parser = argparse.ArgumentParser(description="SAM2 video sequence propagation CLI")
    parser.add_argument("--video-dir", required=True, help="Directory containing ordered RGB frames")
    parser.add_argument("--points", required=True, help='Point coords: "x1,y1 x2,y2"')
    parser.add_argument("--labels", required=True, help='Point labels: "1 0"')
    parser.add_argument("--output-dir", required=True, help="Output mask directory")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--prompt-mask", default="", help="Optional first-frame mask path")
    parser.add_argument("--checkpoint", default="/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _frame_paths(video_dir)
    if args.first_frame < 0 or args.first_frame >= len(frames):
        raise ValueError(f"first-frame {args.first_frame} outside frame count {len(frames)}")

    cv2, np, torch, build_sam2_video_predictor = _load_runtime_dependencies()
    predictor_video_dir, output_names = _prepare_predictor_frames(video_dir, output_dir, cv2)
    points = _parse_points(args.points, np)
    labels = _parse_labels(args.labels, np)
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")

    predictor = build_sam2_video_predictor(args.config, args.checkpoint)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(predictor_video_dir))
        prompt_mode = _add_initial_prompt(
            predictor=predictor,
            state=state,
            first_frame=args.first_frame,
            points=points,
            labels=labels,
            prompt_mask=args.prompt_mask,
            cv2=cv2,
        )

        mask_count = 0
        foreground_pixels = {}
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
            if 1 not in object_ids:
                continue
            obj_index = list(object_ids).index(1)
            mask = (mask_logits[obj_index] > 0.0).cpu().numpy().astype(np.uint8) * 255
            if mask.ndim == 3:
                mask = mask.squeeze()
            out_path = output_dir / output_names[frame_idx]
            cv2.imwrite(str(out_path), mask)
            mask_count += 1
            foreground_pixels[str(frame_idx)] = int((mask > 0).sum())

    print(json.dumps({
        "frame_count": len(frames),
        "mask_count": mask_count,
        "output_dir": str(output_dir),
        "prompt_mode": prompt_mode,
        "foreground_pixels": foreground_pixels,
    }))


if __name__ == "__main__":
    main()
