import json
import math


REQUIRED_KEYS = {
    "aatype",
    "atom_positions",
    "atom_mask",
    "residue_index",
}


def _shape_of_3d(values):
    return (len(values), len(values[0]), len(values[0][0]))


def _shape_of_2d(values):
    return (len(values), len(values[0]))


def _build_mock_example_sample(seq_len: int = 8, atom_num: int = 4):
    """构造一条 examples 样本（mock）。"""
    atom_positions = [
        [[float(i + j + k) / 10.0 for k in range(3)] for j in range(atom_num)]
        for i in range(seq_len)
    ]
    atom_mask = [[float((i + j) % 2) for j in range(atom_num)] for i in range(seq_len)]

    return {
        "aatype": [i % 20 for i in range(seq_len)],
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "residue_index": list(range(seq_len)),
    }


def _build_mock_processed_sample(example_sample: dict):
    processed = {k: json.loads(json.dumps(v)) for k, v in example_sample.items()}
    processed["metadata"] = {
        "source": "mock",
        "seq_len": len(example_sample["aatype"]),
    }
    return processed


def _assert_all_finite_nested(values, name: str):
    if isinstance(values, (int, float)):
        assert math.isfinite(values), f"{name} contains NaN/Inf"
        return
    if isinstance(values, list):
        for x in values:
            _assert_all_finite_nested(x, name)


def test_data_format_matches_examples():
    example_sample = _build_mock_example_sample()
    processed_sample = _build_mock_processed_sample(example_sample)

    assert REQUIRED_KEYS.issubset(example_sample.keys())
    assert REQUIRED_KEYS.issubset(processed_sample.keys())

    # 关键 shape 检查。
    assert len(example_sample["aatype"]) == len(processed_sample["aatype"])
    assert _shape_of_3d(example_sample["atom_positions"]) == _shape_of_3d(processed_sample["atom_positions"])
    assert _shape_of_2d(example_sample["atom_mask"]) == _shape_of_2d(processed_sample["atom_mask"])
    assert len(example_sample["residue_index"]) == len(processed_sample["residue_index"])

    # 关键 dtype 检查（纯 Python mock 下通过类型近似）。
    assert all(isinstance(x, int) for x in processed_sample["aatype"])
    assert all(isinstance(x, int) for x in processed_sample["residue_index"])
    assert all(isinstance(x, float) for row in processed_sample["atom_mask"] for x in row)

    # mask 值域检查。
    unique_vals = {x for row in processed_sample["atom_mask"] for x in row}
    assert unique_vals.issubset({0.0, 1.0}), f"invalid mask values: {unique_vals}"

    # NaN / Inf 检查。
    for key in REQUIRED_KEYS:
        _assert_all_finite_nested(processed_sample[key], key)

    json.dumps(processed_sample["metadata"])
