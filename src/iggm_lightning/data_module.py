# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Lightning DataModule for local processed SAbDab records."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from IgGM.data.convert_to_example_format import convert_entry_to_sample
from IgGM.data.sabdab import load_sabdab_metadata


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class _ProteinSampleDataset(Dataset):
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
            "prot_data_curr": {
                "seq": item["seq"],
                "cord": item["cord"],
                "cmsk": item["cmsk"],
                "mask_design": item["mask_design"],
                "mask_ab": item["mask_ab"],
                "asym_id": item["asym_id"],
                "a-cord": item["a-cord"],
                "a-cmsk": item["a-cmsk"],
                "epitope": item["epitope"],
                "contact": item["contact"],
            },
            "inputs_addi": None,
            "chunk_size": None,
        }


def _batch_one(batch):
    return batch[0]


class ProcessedSabdabDataModule:
    def __init__(
        self,
        metadata_path: str | Path,
        pdb_dir: str | Path,
        batch_size: int = 1,
        num_workers: int = 0,
        split_cfg: Optional[SplitConfig] = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.pdb_dir = Path(pdb_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_cfg = split_cfg or SplitConfig()
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        entries = meta.get("entries", [])
        if not entries:
            tsv_path = self.metadata_path.parents[2] / "metadata" / "sabdab.tsv"
            if tsv_path.exists():
                entries = load_sabdab_metadata(tsv_path)
                sample_ids = set(meta.get("sample_ids", []))
                if sample_ids:
                    entries = [e for e in entries if e.get("pdb_id") in sample_ids]
        items = []
        for entry in entries:
            if str(entry.get("light_chain_id", "")).strip() in {"0", "None", "none", ""}:
                entry = {**entry, "light_chain_id": None}
            pdb_id = entry.get("pdb_id")
            if not pdb_id:
                continue
            pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                continue
            sample = convert_entry_to_sample(entry, pdb_path)
            if sample is None:
                continue
            c = sample["complex"]
            items.append(
                {
                    "seq": c["seq"],
                    "cord": c["cord"],
                    "cmsk": c["cmsk"],
                    "mask_design": c["mask_design"],
                    "mask_ab": c["mask_ab"],
                    "asym_id": c["asym_id"],
                    "a-cord": c["a-cord"],
                    "a-cmsk": c["a-cmsk"],
                    "epitope": c["epitope"],
                    "contact": None,
                }
            )
        if not items:
            raise RuntimeError(f"No valid samples found from {self.metadata_path} and {self.pdb_dir}")
        rnd = random.Random(self.split_cfg.seed)
        rnd.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * self.split_cfg.train_ratio))
        n_val = max(1, int(n * self.split_cfg.val_ratio)) if n > 2 else 0
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:] if (n_train + n_val) < n else items[-1:]

        self.train_ds = _ProteinSampleDataset(train_items)
        self.val_ds = _ProteinSampleDataset(val_items if val_items else train_items[:1])
        self.test_ds = _ProteinSampleDataset(test_items if test_items else train_items[:1])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=_batch_one)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=_batch_one)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=_batch_one)
