"""Annotation dataset stages: QA, review pack, and COCO export."""

import base64
import html
import json
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.input_source import rgb_frame_dir
from pipeline.manifest import load_manifest_for_config
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class MaskStats:
    frame: str
    width: int
    height: int
    area: int
    bbox_xyxy: list[int] | None
    touches_edge: bool


def _stage_input(
    config: PipelineConfig,
    context: StageContext | None,
    stage_name: str,
) -> Path | None:
    if context and context.data and context.data.get_input(stage_name):
        return context.input(stage_name)
    manifest = load_manifest_for_config(config)
    output = manifest.get_output_dir(stage_name)
    return Path(output) if output else None


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != PNG_SIGNATURE:
            raise StageError(f"Not a PNG file: {path}")
        length = struct.unpack(">I", f.read(4))[0]
        chunk_type = f.read(4)
        if chunk_type != b"IHDR" or length < 8:
            raise StageError(f"Invalid PNG header: {path}")
        data = f.read(length)
    width, height = struct.unpack(">II", data[:8])
    return int(width), int(height)


def _png_scanlines(path: Path) -> tuple[int, int, int, bytes]:
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(PNG_SIGNATURE):
        raise StageError(f"Not a PNG file: {path}")

    offset = len(PNG_SIGNATURE)
    width = height = color_type = bit_depth = None
    idat = bytearray()
    while offset < len(raw):
        if offset + 8 > len(raw):
            break
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        chunk_type = raw[offset + 4:offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        data = raw[data_start:data_end]
        offset = data_end + 4
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
            bit_depth = data[8]
            color_type = data[9]
        elif chunk_type == b"IDAT":
            idat.extend(data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or color_type is None or bit_depth != 8:
        raise StageError(f"Unsupported PNG mask format: {path}")
    if color_type not in (0, 2, 4, 6):
        raise StageError(f"Unsupported PNG color type {color_type}: {path}")

    decompressed = zlib.decompress(bytes(idat))
    return int(width), int(height), int(color_type), decompressed


def _channels_for_color_type(color_type: int) -> int:
    return {0: 1, 2: 3, 4: 2, 6: 4}[color_type]


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_scanline(filter_type: int, row: bytearray, prev: bytearray, bpp: int) -> bytearray:
    out = bytearray(row)
    if filter_type == 0:
        return out
    for i in range(len(out)):
        left = out[i - bpp] if i >= bpp else 0
        up = prev[i] if prev else 0
        up_left = prev[i - bpp] if prev and i >= bpp else 0
        if filter_type == 1:
            out[i] = (out[i] + left) & 0xFF
        elif filter_type == 2:
            out[i] = (out[i] + up) & 0xFF
        elif filter_type == 3:
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            out[i] = (out[i] + _paeth(left, up, up_left)) & 0xFF
        else:
            raise StageError(f"Unsupported PNG filter type: {filter_type}")
    return out


def _mask_stats(path: Path) -> MaskStats:
    width, height, color_type, data = _png_scanlines(path)
    channels = _channels_for_color_type(color_type)
    stride = width * channels
    bpp = channels
    pos = 0
    prev = bytearray(stride)
    area = 0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    for y in range(height):
        if pos >= len(data):
            raise StageError(f"PNG scanline data ended early: {path}")
        filter_type = data[pos]
        pos += 1
        row = bytearray(data[pos:pos + stride])
        pos += stride
        row = _unfilter_scanline(filter_type, row, prev, bpp)
        prev = row
        for x in range(width):
            value = row[x * channels]
            if value > 0:
                area += 1
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    bbox = [min_x, min_y, max_x + 1, max_y + 1] if area else None
    touches_edge = bool(bbox and (bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= width or bbox[3] >= height))
    return MaskStats(
        frame=path.name,
        width=width,
        height=height,
        area=area,
        bbox_xyxy=bbox,
        touches_edge=touches_edge,
    )


def _bbox_iou(a: list[int] | None, b: list[int] | None) -> float | None:
    if not a or not b:
        return None
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return None
    return inter / union


def _bbox_center_shift(a: list[int] | None, b: list[int] | None, width: int, height: int) -> float | None:
    if not a or not b:
        return None
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    diagonal = (width ** 2 + height ** 2) ** 0.5
    if diagonal <= 0:
        return None
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diagonal


def _review_status_path(mask_qa_dir: Path) -> Path:
    return mask_qa_dir / "review_status.json"


def _read_review_status(mask_qa_dir: Path) -> dict:
    path = _review_status_path(mask_qa_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _relpath(from_dir: Path, target: Path) -> str:
    import os

    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def _coco_bbox(box: list[int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def _mask_to_coco_rle(path: Path) -> dict:
    width, height, color_type, data = _png_scanlines(path)
    channels = _channels_for_color_type(color_type)
    stride = width * channels
    bpp = channels
    pos = 0
    prev = bytearray(stride)
    counts: list[int] = []
    current_value = 0
    run_length = 0

    rows = []
    for _ in range(height):
        if pos >= len(data):
            raise StageError(f"PNG scanline data ended early: {path}")
        filter_type = data[pos]
        pos += 1
        row = bytearray(data[pos:pos + stride])
        pos += stride
        row = _unfilter_scanline(filter_type, row, prev, bpp)
        prev = row
        rows.append(row)

    for x in range(width):
        for y in range(height):
            value = 1 if rows[y][x * channels] > 0 else 0
            if value == current_value:
                run_length += 1
            else:
                counts.append(run_length)
                current_value = value
                run_length = 1
    counts.append(run_length)
    return {"size": [height, width], "counts": counts}


def _effective_clip_size(frame_count: int, clip_size: int | None) -> tuple[int, str]:
    if clip_size:
        return max(1, int(clip_size)), "configured"
    if frame_count <= 0:
        return 1, "adaptive"
    target_clip_count = min(10, frame_count)
    return max(1, (frame_count + target_clip_count - 1) // target_clip_count), "adaptive"


def _split_by_clips(frames: list[dict], clip_size: int | None, train_ratio: float) -> dict[str, str]:
    clip_size, _ = _effective_clip_size(len(frames), clip_size)
    train_ratio = max(0.0, min(1.0, float(train_ratio)))
    clips = [frames[i:i + clip_size] for i in range(0, len(frames), clip_size)]
    if not clips:
        return {}
    train_clip_count = int(len(clips) * train_ratio)
    if len(clips) == 1:
        train_clip_count = 1
    else:
        train_clip_count = min(len(clips) - 1, max(1, train_clip_count))

    split_by_frame = {}
    for clip_index, clip in enumerate(clips):
        split = "train" if clip_index < train_clip_count else "valid"
        for row in clip:
            split_by_frame[row["frame"]] = split
    return split_by_frame


def _state_from_review(row: dict, review_frames: dict) -> str:
    return review_frames.get(row["frame"], {}).get("state", row["state"])


def _svg_image_href(image_rel: str, output_path: Path) -> str:
    candidate = Path(image_rel)
    if not candidate.is_absolute():
        candidate = output_path.parent / candidate
    if not candidate.exists():
        return image_rel
    mime = "image/jpeg" if candidate.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _draw_box_svg(
    image_rel: str,
    output_path: Path,
    box: list[int],
    width: int,
    height: int,
    label: str,
    state: str = "accepted",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = [int(v) for v in box]
    color = {
        "accepted": "#00d4aa",
        "suspect": "#f0a030",
        "rejected": "#ff4d5a",
    }.get(state, "#00d4aa")
    stroke_width = max(2, int(max(width, height) / 320))
    text = html.escape(label)
    image_href = html.escape(_svg_image_href(image_rel, output_path), quote=True)
    output_path.write_text(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <image href="{image_href}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>
  <rect x="{x1}" y="{y1}" width="{max(1, x2 - x1)}" height="{max(1, y2 - y1)}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>
  <rect x="{x1}" y="{max(0, y1 - 18)}" width="{max(80, len(text) * 7 + 12)}" height="18" fill="{color}"/>
  <text x="{x1 + 4}" y="{max(12, y1 - 5)}" fill="#ffffff" font-size="12" font-family="monospace">{text}</text>
</svg>
""", encoding="utf-8")


def _write_contact_sheet_svg(items: list[dict], output_path: Path, columns: int = 5, thumb_width: int = 220) -> None:
    if not items:
        return
    rows = (len(items) + columns - 1) // columns
    thumb_height = int(thumb_width * 0.75)
    width = columns * thumb_width
    height = rows * (thumb_height + 28)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#07070c"/>',
    ]
    for idx, item in enumerate(items):
        col = idx % columns
        row = idx // columns
        x = col * thumb_width
        y = row * (thumb_height + 28)
        color = {
            "accepted": "#00d4aa",
            "suspect": "#f0a030",
            "rejected": "#ff4d5a",
        }.get(item.get("state"), "#8888a0")
        href = html.escape(item["preview_rel"], quote=True)
        frame = html.escape(item["frame"])
        state = html.escape(item.get("state", ""))
        parts.append(f'<a href="{href}"><image href="{href}" x="{x}" y="{y}" width="{thumb_width}" height="{thumb_height}" preserveAspectRatio="xMidYMid meet"/></a>')
        parts.append(f'<rect x="{x}" y="{y}" width="{thumb_width}" height="{thumb_height}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 6}" y="{y + thumb_height + 18}" fill="{color}" font-size="12" font-family="monospace">{frame} {state}</text>')
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


@register_stage("mask_qa")
class MaskQaStage(BaseStage):
    name = "mask_qa"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = _stage_input(config, context, "sam2_video_propagation")
        if not video_dir:
            raise StageError("No sam2_video_propagation output found")
        masks_dir = video_dir / "masks"
        self.check_input_path(str(masks_dir), "SAM2 propagated masks directory")
        mask_files = sorted(masks_dir.glob("*.png"))
        if not mask_files:
            raise StageError(f"No propagated mask PNG files found in {masks_dir}")

        min_area = max(1, int(config.detection_dataset.min_box_area))
        max_area_ratio = 0.90
        min_area_ratio = 0.0002
        min_iou = 0.15
        max_center_shift = 0.28

        frames = []
        accepted = []
        rejected = []
        suspect = []
        previous_bbox = None
        previous_stats = None
        for mask_file in mask_files:
            stats = _mask_stats(mask_file)
            image_area = stats.width * stats.height
            flags: list[str] = []
            state = "accepted"

            if stats.area < min_area or stats.area < image_area * min_area_ratio:
                flags.append("mask_area_too_small")
                state = "rejected"
            if stats.area > image_area * max_area_ratio:
                flags.append("mask_area_too_large")
                state = "rejected"
            if stats.touches_edge:
                flags.append("mask_touches_image_edge")
                if state != "rejected":
                    state = "suspect"
            if previous_bbox and stats.bbox_xyxy:
                iou = _bbox_iou(previous_bbox, stats.bbox_xyxy)
                shift = _bbox_center_shift(previous_bbox, stats.bbox_xyxy, stats.width, stats.height)
                if iou is not None and iou < min_iou:
                    flags.append("adjacent_bbox_iou_low")
                    if state != "rejected":
                        state = "suspect"
                if shift is not None and shift > max_center_shift:
                    flags.append("bbox_jump_too_large")
                    if state != "rejected":
                        state = "suspect"
            else:
                iou = None
                shift = None

            row = {
                "frame": stats.frame,
                "mask": str(mask_file),
                "width": stats.width,
                "height": stats.height,
                "area": stats.area,
                "area_ratio": stats.area / image_area if image_area else 0,
                "bbox_xyxy": stats.bbox_xyxy,
                "touches_edge": stats.touches_edge,
                "previous_frame": previous_stats.frame if previous_stats else None,
                "previous_iou": iou,
                "center_shift_ratio": shift,
                "flags": flags,
                "state": state,
            }
            frames.append(row)
            if state == "accepted":
                accepted.append(stats.frame)
            elif state == "rejected":
                rejected.append(stats.frame)
            else:
                suspect.append(stats.frame)
            if stats.bbox_xyxy:
                previous_bbox = stats.bbox_xyxy
                previous_stats = stats

        report = {
            "task": config.task,
            "stage": self.name,
            "source_masks": str(masks_dir),
            "rules": {
                "min_box_area": min_area,
                "min_area_ratio": min_area_ratio,
                "max_area_ratio": max_area_ratio,
                "min_adjacent_iou": min_iou,
                "max_center_shift_ratio": max_center_shift,
            },
            "summary": {
                "total": len(frames),
                "accepted": len(accepted),
                "suspect": len(suspect),
                "rejected": len(rejected),
            },
            "frames": frames,
        }
        _write_json(output_dir / "qa_report.json", report)
        _write_json(_review_status_path(output_dir), {
            "task": config.task,
            "source": "mask_qa",
            "frames": {
                row["frame"]: {
                    "state": row["state"],
                    "flags": row["flags"],
                    "manual": False,
                }
                for row in frames
            },
        })
        return output_dir


@register_stage("review_pack")
class ReviewPackStage(BaseStage):
    name = "review_pack"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_qa_dir = _stage_input(config, context, "mask_qa")
        if not mask_qa_dir:
            raise StageError("No mask_qa output found")
        report_path = mask_qa_dir / "qa_report.json"
        self.check_input_path(str(report_path), "Mask QA report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rgb_dir = rgb_frame_dir(config)
        masks_source = Path(report["source_masks"])
        preview_out = output_dir / "preview"
        preview_out.mkdir(exist_ok=True)
        split_by_frame = _split_by_clips(
            report["frames"],
            config.detection_dataset.clip_size,
            config.detection_dataset.train_ratio,
        )
        items = []
        for row in report["frames"]:
            split_name = split_by_frame.get(row["frame"], "train")
            image_path = rgb_dir / row["frame"]
            mask_path = masks_source / row["frame"]
            preview_path = preview_out / f"{Path(row['frame']).stem}.svg"
            if row.get("bbox_xyxy") and image_path.exists():
                _draw_box_svg(
                    image_rel=_relpath(preview_path.parent, image_path),
                    output_path=preview_path,
                    box=row["bbox_xyxy"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    label=f"{config.detection_dataset.class_name} {row['frame']}",
                    state=row["state"],
                )
            item = {
                **row,
                "image": str(image_path) if image_path.exists() else None,
                "mask": str(mask_path) if mask_path.exists() else None,
                "image_rel": _relpath(output_dir, image_path) if image_path.exists() else None,
                "mask_rel": _relpath(output_dir, mask_path) if mask_path.exists() else None,
                "preview_rel": _relpath(output_dir, preview_path) if preview_path.exists() else None,
                "image_fallback_rel": _relpath(output_dir, image_path) if image_path.exists() else None,
                "mask_fallback_rel": _relpath(output_dir, mask_path) if mask_path.exists() else None,
            }
            items.append(item)

        html = _render_review_html(config.task, report["summary"], items)
        (output_dir / "index.html").write_text(html, encoding="utf-8")
        _write_json(output_dir / "review_pack.json", {
            "task": config.task,
            "qa_report": str(report_path),
            "review_status": str(_review_status_path(mask_qa_dir)),
            "summary": report["summary"],
            "items": items,
        })
        return output_dir


def _render_review_html(task: str, summary: dict, items: list[dict]) -> str:
    cards = []
    for item in items:
        flags = ", ".join(item.get("flags") or []) or "-"
        image = html.escape(item.get("image_rel") or "", quote=True)
        mask = html.escape(item.get("mask_rel") or "", quote=True)
        preview = html.escape(item.get("preview_rel") or "", quote=True)
        image_fallback = html.escape(item.get("image_fallback_rel") or "", quote=True)
        mask_fallback = html.escape(item.get("mask_fallback_rel") or "", quote=True)
        frame = html.escape(item["frame"])
        state = html.escape(item["state"])
        hidden = ' data-default-hidden="1"' if item["state"] == "accepted" else ""
        image_html = f'<img src="{image}" data-fallback="{image_fallback}" alt="{frame} RGB">' if image else '<div class="missing">No image</div>'
        mask_html = f'<img src="{mask}" data-fallback="{mask_fallback}" alt="{frame} mask">' if mask else '<div class="missing">No mask</div>'
        preview_html = f'<img src="{preview}" alt="{frame} YOLO box">' if preview else '<div class="missing">No preview</div>'
        cards.append(f"""
        <article class="frame {state}" data-frame="{frame}" data-state="{state}"{hidden}>
          <header><strong>{frame}</strong><span>{state}</span></header>
          <div class="pair">{image_html}{mask_html}{preview_html}</div>
          <dl>
            <dt>bbox</dt><dd>{html.escape(str(item.get('bbox_xyxy')))}</dd>
            <dt>area</dt><dd>{html.escape(str(item.get('area')))}</dd>
            <dt>reason</dt><dd>{html.escape(flags)}</dd>
          </dl>
        </article>
        """)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{task} review pack</title>
  <style>
    body {{ margin:0; background:#07070c; color:#e4e4ec; font:13px system-ui,sans-serif; }}
    main {{ padding:24px; }}
    h1 {{ font-size:20px; margin:0 0 4px; }}
    .summary {{ color:#8888a0; margin-bottom:20px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }}
    .frame {{ border:1px solid #1e1e33; border-radius:4px; background:#0d0d1a; padding:12px; }}
    .frame.suspect {{ border-color:#f0a030; }}
    .frame.rejected {{ border-color:#ff4d5a; opacity:.72; }}
    header {{ display:flex; justify-content:space-between; margin-bottom:8px; }}
    header span {{ color:#00d4aa; font-family:monospace; }}
    .suspect header span {{ color:#f0a030; }}
    .rejected header span {{ color:#ff4d5a; }}
    .pair {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; align-items:start; }}
    img {{ width:100%; background:#050510; border:1px solid #1e1e33; object-fit:contain; }}
    .missing {{ min-height:80px; display:grid; place-items:center; color:#5a5a72; border:1px dashed #1e1e33; }}
    dl {{ display:grid; grid-template-columns:52px 1fr; gap:4px 8px; color:#8888a0; font-family:monospace; font-size:11px; }}
    dt {{ color:#5a5a72; }}
    .toolbar {{ display:flex; gap:8px; margin:12px 0 20px; }}
    input {{ background:#050510; border:1px solid #1e1e33; color:#e4e4ec; padding:8px; border-radius:4px; }}
    button {{ background:#111122; border:1px solid #1e1e33; color:#e4e4ec; padding:8px 10px; border-radius:4px; cursor:pointer; }}
    .hidden {{ display:none; }}
  </style>
</head>
<body>
  <main>
    <h1>{task} review pack</h1>
    <p class="summary">total {summary.get('total', 0)} / accepted {summary.get('accepted', 0)} / suspect {summary.get('suspect', 0)} / rejected {summary.get('rejected', 0)}</p>
    <div class="toolbar">
      <input id="search" placeholder="搜索 frame，例如 00042">
      <button onclick="setMode('issues')">仅异常</button>
      <button onclick="setMode('all')">全部</button>
    </div>
    <section class="grid">{''.join(cards)}</section>
  </main>
  <script>
    let mode = 'issues';
    function applyFilter() {{
      const q = document.getElementById('search').value.trim();
      document.querySelectorAll('.frame').forEach(el => {{
        const matchesSearch = !q || el.dataset.frame.includes(q);
        const matchesMode = mode === 'all' || el.dataset.state !== 'accepted';
        el.classList.toggle('hidden', !(matchesSearch && matchesMode));
      }});
    }}
    function setMode(next) {{ mode = next; applyFilter(); }}
    document.getElementById('search').addEventListener('input', applyFilter);
    document.querySelectorAll('img[data-fallback]').forEach(img => {{
      img.addEventListener('error', () => {{
        const fallback = img.dataset.fallback;
        if (fallback && img.src !== fallback) {{
          img.removeAttribute('data-fallback');
          img.src = fallback;
        }}
      }}, {{ once: true }});
    }});
    applyFilter();
  </script>
</body>
</html>
"""


@register_stage("detection_dataset_export")
class DetectionDatasetExportStage(BaseStage):
    name = "detection_dataset_export"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_qa_dir = _stage_input(config, context, "mask_qa")
        if not mask_qa_dir:
            raise StageError("No mask_qa output found")
        report_path = mask_qa_dir / "qa_report.json"
        self.check_input_path(str(report_path), "Mask QA report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        review = _read_review_status(mask_qa_dir)
        review_frames = review.get("frames", {})
        rgb_dir = rgb_frame_dir(config)
        masks_dir = Path(report["source_masks"])
        split_names = ("train", "valid")
        split_dirs = {name: output_dir / name for name in split_names}
        masks_out = output_dir / "masks"
        preview_out = output_dir / "preview"
        contact_sheet_path = output_dir / "contact_sheet.svg"
        for stale_dir in (
            split_dirs["train"],
            split_dirs["valid"],
            output_dir / "images",
            output_dir / "labels",
            masks_out,
            preview_out,
            output_dir / "test",
        ):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        if contact_sheet_path.exists():
            contact_sheet_path.unlink()
        old_annotations_path = output_dir / "annotations.json"
        if old_annotations_path.exists():
            old_annotations_path.unlink()
        dataset_yaml_path = output_dir / "dataset.yaml"
        if dataset_yaml_path.exists():
            dataset_yaml_path.unlink()
        for split_dir in split_dirs.values():
            split_dir.mkdir(exist_ok=True)
        masks_out.mkdir(exist_ok=True)
        preview_out.mkdir(exist_ok=True)

        coco_by_split = {
            name: {
                "images": [],
                "annotations": [],
                "next_image_id": 1,
                "next_annotation_id": 1,
            }
            for name in split_names
        }
        skipped = []
        contact_items = []
        split_by_frame = _split_by_clips(
            report["frames"],
            config.detection_dataset.clip_size,
            config.detection_dataset.train_ratio,
        )
        effective_clip_size, clip_size_mode = _effective_clip_size(
            len(report["frames"]),
            config.detection_dataset.clip_size,
        )
        for row in report["frames"]:
            frame = row["frame"]
            split_name = split_by_frame.get(frame, "train")
            split_out = split_dirs[split_name]
            split_coco = coco_by_split[split_name]
            review_state = review_frames.get(frame, {}).get("state", row["state"])
            if review_state == "rejected":
                skipped.append({"image": frame, "reason": "rejected", "split": split_name})
                continue
            if not row.get("bbox_xyxy"):
                skipped.append({"image": frame, "reason": "missing_bbox", "split": split_name})
                continue

            image_path = rgb_dir / frame
            mask_path = masks_dir / frame
            if not image_path.exists():
                skipped.append({"image": frame, "reason": "missing_image", "split": split_name})
                continue
            if not mask_path.exists():
                skipped.append({"image": frame, "reason": "missing_mask", "split": split_name})
                continue

            dst_image = split_out / frame
            dst_mask = masks_out / frame
            if config.detection_dataset.copy_images:
                shutil.copy2(image_path, dst_image)
            else:
                dst_image = image_path
            shutil.copy2(mask_path, dst_mask)

            width = int(row["width"])
            height = int(row["height"])
            coco_bbox = _coco_bbox(row["bbox_xyxy"])
            segmentation = _mask_to_coco_rle(mask_path)
            preview_path = preview_out / f"{Path(frame).stem}.svg"
            _draw_box_svg(
                image_rel=_relpath(preview_path.parent, dst_image),
                output_path=preview_path,
                box=row["bbox_xyxy"],
                width=width,
                height=height,
                label=f"{config.detection_dataset.class_name} {frame}",
                state=review_state,
            )
            contact_items.append({
                "frame": frame,
                "state": review_state,
                "preview_rel": _relpath(output_dir, preview_path),
            })
            image_id = split_coco["next_image_id"]
            annotation_id = split_coco["next_annotation_id"]
            split_coco["images"].append({
                "id": image_id,
                "file_name": dst_image.name,
                "width": width,
                "height": height,
            })
            split_coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(config.detection_dataset.class_id),
                "segmentation": segmentation,
                "area": int(coco_bbox[2] * coco_bbox[3]),
                "bbox": coco_bbox,
                "iscrowd": 0,
                "source": {
                    "image": str(dst_image),
                    "mask": str(dst_mask),
                    "preview": str(preview_path),
                    "frame": frame,
                    "qa_state": row["state"],
                    "review_state": review_state,
                    "flags": row.get("flags", []),
                },
            })
            split_coco["next_image_id"] += 1
            split_coco["next_annotation_id"] += 1

        _write_contact_sheet_svg(contact_items, contact_sheet_path)
        categories = [{
            "id": int(config.detection_dataset.class_id),
            "name": config.detection_dataset.class_name,
            "supercategory": "object",
        }]
        for split_name in split_names:
            split_coco = coco_by_split[split_name]
            split_skipped = [item for item in skipped if item.get("split") == split_name]
            _write_json(split_dirs[split_name] / "_annotations.coco.json", {
                "info": {
                    "description": f"{config.task} {split_name} annotation dataset",
                    "version": "1.0",
                },
                "licenses": [],
                "images": split_coco["images"],
                "annotations": split_coco["annotations"],
                "categories": categories,
                "metadata": {
                    "task": config.task,
                    "format": "coco",
                    "source": "sam2_masks",
                    "class_id": int(config.detection_dataset.class_id),
                    "class_name": config.detection_dataset.class_name,
                    "count": len(split_coco["annotations"]),
                    "contact_sheet": str(contact_sheet_path),
                    "skipped": split_skipped,
                    "split": split_name,
                    "clip_size": effective_clip_size,
                    "clip_size_mode": clip_size_mode,
                    "train_ratio": float(config.detection_dataset.train_ratio),
                },
            })
        return output_dir
