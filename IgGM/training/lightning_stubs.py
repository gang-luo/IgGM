from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


try:
    import lightning as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L


class DebugDataModule(L.LightningDataModule):
    """Minimal random dataset used by debug/training templates."""

    def __init__(
        self,
        input_dim: int = 16,
        train_size: int = 256,
        val_size: int = 64,
        test_size: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _build_dataset(self, size: int) -> TensorDataset:
        x = torch.randn(size, self.input_dim)
        y = x.sum(dim=1, keepdim=True)
        return TensorDataset(x, y)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(self.train_size)
            self.val_dataset = self._build_dataset(self.val_size)
        if stage in (None, "test", "predict"):
            self.test_dataset = self._build_dataset(self.test_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


class DebugLightningModule(L.LightningModule):
    """Simple regressor for smoke-testing entrypoint scripts."""

    def __init__(self, input_dim: int = 16, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _common_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._common_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._common_step(batch, "test")

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        pred = self(x)
        return {
            "sequence": [f"pred_{batch_idx}_{i}:{value.item():.4f}" for i, value in enumerate(pred[:, 0])],
            "structure": [f"MODEL {batch_idx}-{i}\nREMARK {value.item():.4f}\nENDMDL\n" for i, value in enumerate(pred[:, 0])],
            "target": y[:, 0].detach().cpu().tolist(),
            "prediction": pred[:, 0].detach().cpu().tolist(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
