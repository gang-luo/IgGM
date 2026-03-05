#!/usr/bin/env python3
"""Prepare training samples into torch serialized records."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in lightweight envs
    torch = None

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "UNK": "X",
}

CHOTHIA_RANGES = {
    "cdr_L1": (24, 34),
    "cdr_L2": (50, 56),
    "cdr_L3": (89, 97),
    "cdr_H1": (26, 32),
    "cdr_H2": (52, 56),
    "cdr_H3": (95, 102),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset samples from SAbDab or local mock data")
    parser.add_argument("--raw_root", type=Path, default=Path("data/raw"), help="Root dir for raw SAbDab-like data")
    parser.add_argument("--out_root", type=Path, default=Path("data/processed"), help="Root output dir")
    parser.add_argument("--dataset_name", type=str, required=True, help="Output dataset folder name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker threads for sample conversion")
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Reserved switch for downloading data (not required when local raw data exists)",
    )
    return parser.parse_args()


def read_fasta_sequences(path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_header: Optional[str] = None
    chunks: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                sequences[current_header] = "".join(chunks)
            current_header = line[1:].strip() or f"seq_{len(sequences)}"
            chunks = []
        else:
            chunks.append(line)
    if current_header is not None:
        sequences[current_header] = "".join(chunks)
    return sequences


def find_sabdab_metadata(raw_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    patterns = ["*sabdab*.csv", "*sabdab*.tsv", "*metadata*.csv", "*metadata*.tsv"]
    for pattern in patterns:
        candidates.extend(raw_root.rglob(pattern))
    return sorted(set(candidates))[0] if candidates else None


def detect_mode(raw_root: Path) -> str:
    metadata = find_sabdab_metadata(raw_root)
    has_structure = any(raw_root.rglob("*.pdb"))
    if metadata is not None and has_structure:
        return "real"
    return "mock"


def load_metadata_rows(metadata_path: Path) -> List[Dict[str, str]]:
    delimiter = "\t" if metadata_path.suffix.lower() == ".tsv" else ","
    with metadata_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delimiter)
        return [dict(row) for row in reader]


def build_pdb_index(raw_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for pdb_file in raw_root.rglob("*.pdb"):
        index.setdefault(pdb_file.stem.lower(), pdb_file)
    return index


def choose_first(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value:
            return value.strip()
    return None


def parse_chain_from_sample_id(sample_id: str) -> Dict[str, Optional[str]]:
    parts = sample_id.split("_")
    chain_ids = {"H": None, "L": None, "A": None}
    if len(parts) >= 4:
        chain_ids["H"] = parts[1] if parts[1] != "NA" else None
        chain_ids["L"] = parts[2] if parts[2] != "NA" else None
        chain_ids["A"] = parts[3] if parts[3] != "NA" else None
    return chain_ids


def parse_chain_ids(entry: Dict[str, object]) -> Dict[str, Optional[str]]:
    row = entry.get("raw_row", {})
    chain_ids = parse_chain_from_sample_id(str(entry["sample_id"]))
    if isinstance(row, dict):
        chain_ids["H"] = chain_ids["H"] or choose_first(row, ["Hchain", "heavy_chain", "heavy", "chain_h"])
        chain_ids["L"] = chain_ids["L"] or choose_first(row, ["Lchain", "light_chain", "light", "chain_l"])
        chain_ids["A"] = chain_ids["A"] or choose_first(
            row, ["antigen_chain", "antigen", "chain_a", "antigen_chain_id"]
        )
    return chain_ids


def parse_pdb_sequences(pdb_path: Path) -> Dict[str, List[Tuple[int, str]]]:
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            icode = line[26].strip()
            residue_key = (chain_id, resseq, icode)
            if residue_key in seen:
                continue
            seen.add(residue_key)
            aa = AA3_TO_1.get(resname, "X")
            chain_residues.setdefault(chain_id, []).append((resseq, aa))
    return chain_residues


def build_cdr_indices(chain_residues: List[Tuple[int, str]], cdr_names: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    cdr_pdb: Dict[str, List[int]] = {}
    cdr_seq: Dict[str, List[int]] = {}
    for cdr_name in cdr_names:
        start, end = CHOTHIA_RANGES[cdr_name]
        pdb_positions: List[int] = []
        seq_positions: List[int] = []
        for seq_idx, (pdb_resseq, _) in enumerate(chain_residues, start=1):
            if start <= pdb_resseq <= end:
                pdb_positions.append(pdb_resseq)
                seq_positions.append(seq_idx)
        cdr_pdb[cdr_name] = pdb_positions
        cdr_seq[cdr_name] = seq_positions
    return cdr_pdb, cdr_seq


def extract_sequences_and_cdr(entry: Dict[str, object]) -> Tuple[Dict[str, str], Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
    pdb_path = Path(str(entry["pdb_path"]))
    chain_ids = parse_chain_ids(entry)
    parsed = parse_pdb_sequences(pdb_path)

    sequences: Dict[str, str] = {}
    cdr_pdb: Dict[str, Dict[str, List[int]]] = {}
    cdr_sequences: Dict[str, Dict[str, List[int]]] = {}

    for role in ("H", "L", "A"):
        chain_id = chain_ids.get(role)
        if not chain_id or chain_id not in parsed:
            continue
        residues = parsed[chain_id]
        sequences[role] = "".join(aa for _, aa in residues)
        if role == "H":
            cdr_pdb["P"] = cdr_pdb.get("P", {})
            cdr_sequences["P"] = cdr_sequences.get("P", {})
            pdb_idx, seq_idx = build_cdr_indices(residues, ["cdr_H1", "cdr_H2", "cdr_H3"])
            cdr_pdb["P"].update(pdb_idx)
            cdr_sequences["P"].update(seq_idx)
        elif role == "L":
            cdr_pdb["P"] = cdr_pdb.get("P", {})
            cdr_sequences["P"] = cdr_sequences.get("P", {})
            pdb_idx, seq_idx = build_cdr_indices(residues, ["cdr_L1", "cdr_L2", "cdr_L3"])
            cdr_pdb["P"].update(pdb_idx)
            cdr_sequences["P"].update(seq_idx)

    return sequences, cdr_pdb, cdr_sequences


def write_processed_fasta(fasta_dir: Path, sample_id: str, sequences: Dict[str, str]) -> Path:
    fasta_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = fasta_dir / f"{sample_id}.fasta"
    lines: List[str] = []
    for tag in ("H", "L", "A"):
        seq = sequences.get(tag)
        if seq:
            lines.append(f">{tag}")
            lines.append(seq)
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fasta_path


def prepare_real_entries(raw_root: Path) -> List[Dict[str, object]]:
    metadata_path = find_sabdab_metadata(raw_root)
    if metadata_path is None:
        return []

    rows = load_metadata_rows(metadata_path)
    pdb_index = build_pdb_index(raw_root)
    entries: List[Dict[str, object]] = []
    for row in rows:
        entry_id = choose_first(row, ["pdb", "pdb_id", "pdb_code", "id", "entry", "name"])
        if not entry_id:
            continue

        pdb_path_raw = choose_first(row, ["pdb_path", "structure_path", "pdb_file", "file"])
        pdb_path: Optional[Path] = None
        if pdb_path_raw:
            candidate = Path(pdb_path_raw)
            pdb_path = candidate if candidate.is_absolute() else (raw_root / candidate)
            if not pdb_path.exists():
                pdb_path = None
        if pdb_path is None:
            pdb_path = pdb_index.get(entry_id.lower())

        if pdb_path is None or not pdb_path.exists():
            continue

        entries.append(
            {
                "sample_id": entry_id,
                "mode": "real",
                "source_metadata": str(metadata_path),
                "pdb_path": str(pdb_path),
                "raw_row": row,
            }
        )
    return entries


def prepare_mock_entries() -> List[Dict[str, object]]:
    pdb_root = Path("examples/pdb.files.native")
    entries: List[Dict[str, object]] = []
    for pdb_path in sorted(pdb_root.glob("*.pdb")):
        entries.append(
            {
                "sample_id": pdb_path.stem,
                "mode": "mock",
                "pdb_path": str(pdb_path),
                "raw_row": {},
            }
        )
    return entries


def finalize_sample(entry: Dict[str, object], fasta_dir: Path) -> Dict[str, object]:
    sequences, cdr_pdb, cdr_sequences = extract_sequences_and_cdr(entry)
    seq_lens = {name: len(seq) for name, seq in sequences.items()}
    fasta_path = write_processed_fasta(fasta_dir, str(entry["sample_id"]), sequences)

    sample = {
        "sample_id": entry["sample_id"],
        "mode": entry["mode"],
        "pdb_path": entry["pdb_path"],
        "fasta_path": str(fasta_path),
        "source_metadata": entry.get("source_metadata"),
        "sequences": sequences,
        "sequence_lengths": seq_lens,
        "cdr_pdb": cdr_pdb,
        "cdr_sequences": cdr_sequences,
        "length_tensor": (
            torch.tensor(list(seq_lens.values()), dtype=torch.int32)
            if torch is not None
            else {"data": list(seq_lens.values()), "shape": [len(seq_lens)], "dtype": "int32"}
        ),
        "raw_row": entry.get("raw_row", {}),
    }
    return sample


def summarize_sample(sample: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {"keys": sorted(sample.keys()), "tensors": {}, "sequences": {}}
    for key, value in sample.items():
        if torch is not None and torch.is_tensor(value):
            summary["tensors"][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict) and {"shape", "dtype"}.issubset(value.keys()):
            summary["tensors"][key] = {"shape": value["shape"], "dtype": value["dtype"]}
    sequences = sample.get("sequences", {})
    if isinstance(sequences, dict):
        summary["sequences"] = {k: len(v) for k, v in sequences.items()}
    return summary


def main() -> None:
    args = parse_args()

    if args.download:
        print("[INFO] --download was provided; automatic download is not implemented, using local files only.")

    mode = detect_mode(args.raw_root)
    if mode == "real":
        entries = prepare_real_entries(args.raw_root)
    else:
        entries = prepare_mock_entries()

    if not entries:
        raise RuntimeError("No entries found in either real mode or mock mode inputs.")

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    if args.limit is not None and args.limit >= 0:
        entries = entries[: args.limit]

    dataset_root = args.out_root / args.dataset_name
    samples_dir = dataset_root / "samples"
    fasta_dir = args.out_root / "fasta"
    samples_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.num_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(lambda item: finalize_sample(item, fasta_dir), entries))

    for idx, sample in enumerate(samples):
        sample_path = samples_dir / f"{idx:06d}_{sample['sample_id']}.pt"
        if torch is not None:
            torch.save(sample, sample_path)
        else:
            with sample_path.open("wb") as fp:
                pickle.dump(sample, fp)

    metadata = {
        "dataset_name": args.dataset_name,
        "mode": mode,
        "sample_count": len(samples),
        "samples_dir": str(samples_dir),
        "fasta_dir": str(fasta_dir),
        "seed": args.seed,
        "limit": args.limit,
        "num_workers": workers,
        "download": bool(args.download),
        "sample_ids": [s["sample_id"] for s in samples],
    }
    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    first_summary = summarize_sample(samples[0])
    print(f"Sample count: {len(samples)}")
    print(f"Output path: {dataset_root}")
    print("First sample summary:")
    print(json.dumps(first_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# python scripts/prepare_data.py --dataset_name sabdab --raw_root notebooks/data/sabdab/metadata --out_root notebooks/data/sabdab/processed