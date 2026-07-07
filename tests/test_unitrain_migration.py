import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unitrain_package_imports_after_migration():
    unitrain = importlib.import_module("unitrain")

    assert hasattr(unitrain, "get_runner")
    assert hasattr(unitrain, "load_config")


def test_unitrain_cli_modules_import_after_migration():
    train = importlib.import_module("cli.train")
    evaluate = importlib.import_module("cli.eval")
    export = importlib.import_module("cli.export")
    predict = importlib.import_module("cli.predict")

    assert callable(train.main)
    assert callable(evaluate.main)
    assert callable(export.main)
    assert callable(predict.main)


def test_unitrain_support_files_are_inside_unified_root():
    assert (ROOT / "envs" / "rfdetr.txt").exists()
    assert (ROOT / "envs" / "ultralytics.txt").exists()
    assert (ROOT / "weights").is_dir()
    assert (ROOT / "run_unitrain.sh").exists()
