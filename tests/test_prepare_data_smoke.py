import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


CANDIDATE_DATASET_MODULES = [
    "IgGM.data.processed_sabdab_dataset",
    "IgGM.data.dataset",
    "IgGM.dataset",
    "dataset",
]


def _load_processed_dataset_class():
    for module_name in CANDIDATE_DATASET_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, "ProcessedSabdabDataset", None)
        if cls is not None:
            return cls
    return None


def test_prepare_data_smoke(tmp_path: Path):
    script_path = Path("scripts/prepare_data.py")
    if not script_path.exists():
        pytest.skip("scripts/prepare_data.py not found in this repository")

    output_dir = tmp_path / "processed"

    cmd = [
        sys.executable,
        str(script_path),
        "--limit",
        "2",
        "--mock",
        "--output_dir",
        str(output_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 0, (
        f"prepare_data failed with code={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    # 输出目录结构 + metadata.json。
    assert output_dir.exists() and output_dir.is_dir()
    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists(), "metadata.json was not generated"

    metadata = json.loads(metadata_path.read_text())
    assert isinstance(metadata, dict)

    dataset_cls = _load_processed_dataset_class()
    if dataset_cls is None:
        pytest.skip("ProcessedSabdabDataset class not found in expected modules")

    dataset = dataset_cls(str(output_dir))
    assert len(dataset) >= 1, "ProcessedSabdabDataset should contain at least one sample"
