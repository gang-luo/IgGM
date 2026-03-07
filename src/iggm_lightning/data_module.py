# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Lightning DataModule for local processed SAbDab records."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from torch.utils.data import DataLoader, Dataset, Sampler

from IgGM.data.sabdab import load_sabdab_metadata


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class _ProteinSampleDataset(Dataset):
    """Lazy dataset that keeps only metadata and sample paths in memory."""

    def __init__(self, items: List[Dict[str, object]], n_steps: int = 200):
        self.items = items
        self.n_steps = n_steps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        step = random.randint(1, self.n_steps)
        return {
            "idx_step": step,
            "prot_id": item["prot_id"],
            "sample_path": item.get("sample_path"),
            "processed_pdb_path": item.get("processed_pdb_path"),
            "pdb_id": item.get("pdb_id"),
            "inputs_addi": None,
            "chunk_size": None,
        }


class ClusterEpochSampler(Sampler[int]):
    """Sample one item per cluster for each epoch."""

    def __init__(self, clusters: List[List[int]], seed: int = 42) -> None:
        self.clusters = [c for c in clusters if c]
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = [rng.choice(cluster) for cluster in self.clusters]
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.clusters)


def _batch_one(batch):
    return batch[0]


def _load_id_set(path: Optional[str | Path]) -> Optional[set[str]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    ids = {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
    return ids if ids else None


def _parse_prot_id(prot_id: str) -> Optional[Dict[str, object]]:
    """Parse split `prot_id` string to a minimal metadata-like entry."""
    parts = prot_id.strip().split("_")
    if len(parts) != 4:
        return None
    pdb_id, heavy_id, light_id, antigen_id = parts
    light = None if light_id.upper() == "NA" else light_id
    return {
        "pdb_id": pdb_id.lower(),
        "heavy_chain_id": heavy_id,
        "light_chain_id": light,
        "antigen_chain_ids": [antigen_id],
        "prot_id": prot_id,
        "file_stem": prot_id,
    }


class ProcessedSabdabDataModule:
    def __init__(
        self,
        metadata_path: str | Path,
        pdb_dir: str | Path,
        batch_size: int = 1,
        num_workers: int = 0,
        split_cfg: Optional[SplitConfig] = None,
        train_ids_path: Optional[str | Path] = None,
        val_ids_path: Optional[str | Path] = None,
        test_ids_path: Optional[str | Path] = None,
        train_cluster_path: Optional[str | Path] = None,
        samples_dir: Optional[str | Path] = None,
        n_steps: int = 200,
    ):
        self.metadata_path = Path(metadata_path)
        self.pdb_dir = Path(pdb_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_cfg = split_cfg or SplitConfig()
        self.train_ids = _load_id_set(train_ids_path)
        self.val_ids = _load_id_set(val_ids_path)
        self.test_ids = _load_id_set(test_ids_path)
        self.train_cluster_path = Path(train_cluster_path) if train_cluster_path else None
        self.samples_dir = Path(samples_dir) if samples_dir else None
        self.n_steps = n_steps
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self._train_sampler: Optional[Sampler[int]] = None

    def _resolve_samples_dir(self, meta: Dict[str, object]) -> Optional[Path]:
        if self.samples_dir is not None:
            return self.samples_dir
        sample_dir = meta.get("samples_dir")
        if sample_dir:
            p = Path(str(sample_dir))
            if p.exists():
                return p
            p2 = self.metadata_path.parents[2] / p.relative_to("data") if str(p).startswith("data/") else None
            if p2 is not None and p2.exists():
                return p2
        candidate = self.metadata_path.parent / "samples"
        return candidate if candidate.exists() else None

    def _load_entries(self) -> List[Dict[str, object]]:
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        entries = meta.get("entries", [])
        if entries:
            return entries

        split_ids: set[str] = set()
        for id_set in (self.train_ids, self.val_ids, self.test_ids):
            if id_set:
                split_ids.update(id_set)
        if split_ids:
            parsed = [_parse_prot_id(x) for x in sorted(split_ids)]
            return [x for x in parsed if x is not None]

        sample_ids = meta.get("sample_ids", [])
        if sample_ids:
            stems = {Path(p).stem for p in self.pdb_dir.glob("*.pdb")}
            parsed = []
            for stem in sorted(stems):
                if not any(stem == sid or stem.startswith(f"{sid}_") for sid in sample_ids):
                    continue
                item = _parse_prot_id(stem)
                if item is not None:
                    parsed.append(item)
            if parsed:
                return parsed

        tsv_path = self.metadata_path.parents[2] / "metadata" / "sabdab.tsv"
        if tsv_path.exists():
            entries = load_sabdab_metadata(tsv_path)
            sample_ids = set(meta.get("sample_ids", []))
            if sample_ids:
                entries = [e for e in entries if e.get("pdb_id") in sample_ids]
        return entries

    def _build_items(self, entries: Iterable[Dict[str, object]], samples_dir: Optional[Path]) -> List[Dict[str, object]]:
        items = []
        sample_index: Dict[str, Path] = {}
        if samples_dir is not None and samples_dir.exists():
            sample_index = {p.stem: p for p in samples_dir.glob("*.pt")}

        for entry in entries:
            prot_id = str(entry.get("prot_id") or "").strip()
            if not prot_id:
                pdb_id = str(entry.get("pdb_id", "")).lower()
                heavy = str(entry.get("heavy_chain_id") or "").strip()
                light = str(entry.get("light_chain_id") or "NA").strip() or "NA"
                if light in {"0", "None", "none"}:
                    light = "NA"
                ag_list = entry.get("antigen_chain_ids") or []
                ag = str(ag_list[0]).strip() if ag_list else ""
                if not pdb_id or not heavy or not ag:
                    continue
                prot_id = f"{pdb_id}_{heavy}_{light}_{ag}"

            sample_path = sample_index.get(prot_id)
            processed_pdb_path = self.pdb_dir / f"{prot_id}.pdb"
            if not processed_pdb_path.exists():
                continue

            items.append(
                {
                    "prot_id": prot_id,
                    "pdb_id": str(entry.get("pdb_id", "")).lower(),
                    "sample_path": str(sample_path) if sample_path is not None else None,
                    "processed_pdb_path": str(processed_pdb_path),
                }
            )
        return items

    def _build_sampler_from_cluster_file(self, train_items: List[Dict[str, object]]) -> Optional[Sampler[int]]:
        if self.train_cluster_path is None or not self.train_cluster_path.exists():
            return None
        id2idx = {x["prot_id"]: i for i, x in enumerate(train_items)}
        clusters: List[List[int]] = []
        for line in self.train_cluster_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            members = parts[-1].split()
            idxs = [id2idx[m] for m in members if m in id2idx]
            if idxs:
                clusters.append(idxs)
        return ClusterEpochSampler(clusters, seed=self.split_cfg.seed) if clusters else None

    def setup(self, stage: Optional[str] = None):
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        samples_dir = self._resolve_samples_dir(meta)
        entries = self._load_entries()
        items = self._build_items(entries, samples_dir)
        if not items:
            raise RuntimeError(f"No valid samples found from {self.metadata_path} and {self.pdb_dir}")

        if self.train_ids is not None:
            train_items = [x for x in items if x["prot_id"] in self.train_ids]
            val_items = [x for x in items if self.val_ids and x["prot_id"] in self.val_ids]
            test_items = [x for x in items if self.test_ids and x["prot_id"] in self.test_ids]
        else:
            rnd = random.Random(self.split_cfg.seed)
            rnd.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * self.split_cfg.train_ratio))
            n_val = max(1, int(n * self.split_cfg.val_ratio)) if n > 2 else 0
            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:] if (n_train + n_val) < n else items[-1:]

        self.train_ds = _ProteinSampleDataset(train_items, n_steps=self.n_steps)
        self.val_ds = _ProteinSampleDataset(val_items if val_items else train_items[:1], n_steps=self.n_steps)
        self.test_ds = _ProteinSampleDataset(test_items if test_items else train_items[:1], n_steps=self.n_steps)
        self._train_sampler = self._build_sampler_from_cluster_file(train_items)

    def train_dataloader(self):
        if self._train_sampler is not None:
            return DataLoader(self.train_ds, batch_size=1, shuffle=False, sampler=self._train_sampler, num_workers=self.num_workers, collate_fn=_batch_one)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=_batch_one)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=_batch_one)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=_batch_one)
