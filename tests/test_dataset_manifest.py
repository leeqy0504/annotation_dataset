import json
from pathlib import Path

from datasets.manifest import DatasetValidationError, prepare_dataset_manifest


def _write_split(root: Path, split: str, image_names: list[str], categories: list[dict]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    images = []
    annotations = []
    for index, image_name in enumerate(image_names, start=1):
        (split_dir / image_name).write_bytes(b"fake image")
        images.append({
            "id": index,
            "file_name": image_name,
            "width": 4,
            "height": 3,
        })
        annotations.append({
            "id": index,
            "image_id": index,
            "category_id": categories[0]["id"],
            "bbox": [0, 0, 2, 2],
            "area": 4,
            "iscrowd": 0,
        })
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps({
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }),
        encoding="utf-8",
    )


def test_prepare_dataset_manifest_writes_roboflow_coco_contract(tmp_path):
    dataset_root = tmp_path / "detection_dataset_export"
    output_dir = tmp_path / "dataset_prepare"
    categories = [{"id": 0, "name": "object", "supercategory": "object"}]
    _write_split(dataset_root, "train", ["000000.png", "000001.png"], categories)
    _write_split(dataset_root, "valid", ["000002.png"], categories)

    manifest = prepare_dataset_manifest(
        dataset_root=dataset_root,
        output_dir=output_dir,
        task_name="mouse_001",
        run_id="run42",
        source_stage="detection_dataset_export",
    )

    manifest_path = output_dir / "dataset_manifest.json"
    assert manifest_path.exists()
    assert manifest["dataset_id"] == "mouse_001:run42:detection_dataset_export"
    assert manifest["format"] == "roboflow_coco"
    assert manifest["root"] == str(dataset_root)
    assert manifest["splits"]["train"]["image_count"] == 2
    assert manifest["splits"]["train"]["annotation_count"] == 2
    assert manifest["splits"]["valid"]["image_count"] == 1
    assert manifest["splits"]["valid"]["annotation_count"] == 1
    assert manifest["categories"] == [{"id": 0, "name": "object"}]
    assert manifest["validation"] == {"status": "passed", "warnings": []}


def test_prepare_dataset_manifest_rejects_missing_referenced_image(tmp_path):
    dataset_root = tmp_path / "detection_dataset_export"
    output_dir = tmp_path / "dataset_prepare"
    categories = [{"id": 0, "name": "object"}]
    _write_split(dataset_root, "train", ["000000.png"], categories)
    _write_split(dataset_root, "valid", ["000001.png"], categories)
    (dataset_root / "valid" / "000001.png").unlink()

    try:
        prepare_dataset_manifest(
            dataset_root=dataset_root,
            output_dir=output_dir,
            task_name="mouse_001",
            run_id=None,
            source_stage="detection_dataset_export",
        )
    except DatasetValidationError as exc:
        assert "Referenced image not found" in str(exc)
        assert "000001.png" in str(exc)
    else:
        raise AssertionError("prepare_dataset_manifest should reject missing images")
