"""Dataset manifest generation for pipeline-to-training handoff."""

import json
from pathlib import Path
from typing import Any


class DatasetValidationError(Exception):
    """Raised when an exported dataset does not satisfy the training contract."""


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _category_key(categories: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [(int(item["id"]), str(item["name"])) for item in categories]


def _normalized_categories(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": int(item["id"]), "name": str(item["name"])} for item in categories]


def _validate_split(dataset_root: Path, split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_dir = dataset_root / split_name
    annotations_path = split_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise DatasetValidationError(f"Missing annotation file for split '{split_name}': {annotations_path}")

    coco = _read_json(annotations_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not images:
        raise DatasetValidationError(f"Split '{split_name}' has no images")
    if not categories:
        raise DatasetValidationError(f"Split '{split_name}' has no categories")

    for image in images:
        file_name = image.get("file_name")
        image_path = split_dir / str(file_name)
        if not image_path.exists():
            raise DatasetValidationError(f"Referenced image not found for split '{split_name}': {image_path}")

    split_manifest = {
        "images_dir": str(split_dir),
        "annotations": str(annotations_path),
        "image_count": len(images),
        "annotation_count": len(annotations),
    }
    return split_manifest, categories


def prepare_dataset_manifest(
    *,
    dataset_root: Path,
    output_dir: Path,
    task_name: str,
    run_id: str | None,
    source_stage: str,
) -> dict[str, Any]:
    """Validate Roboflow-style COCO output and write dataset_manifest.json."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    split_manifests: dict[str, dict[str, Any]] = {}
    category_sets: list[list[tuple[int, str]]] = []
    first_categories: list[dict[str, Any]] | None = None
    for split_name in ("train", "valid"):
        split_manifest, categories = _validate_split(dataset_root, split_name)
        split_manifests[split_name] = split_manifest
        category_sets.append(_category_key(categories))
        if first_categories is None:
            first_categories = categories

    if len({tuple(category_set) for category_set in category_sets}) != 1:
        raise DatasetValidationError("Inconsistent categories between train and valid splits")

    dataset_id = f"{task_name}:{run_id or 'default'}:{source_stage}"
    manifest = {
        "dataset_id": dataset_id,
        "format": "roboflow_coco",
        "root": str(dataset_root),
        "splits": split_manifests,
        "categories": _normalized_categories(first_categories or []),
        "source_stage": source_stage,
        "validation": {
            "status": "passed",
            "warnings": [],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
