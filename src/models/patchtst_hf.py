"""PatchTST forecaster via HuggingFace ``transformers.PatchTSTForPrediction``.

darts doesn't ship PatchTST, so we wire the official HuggingFace
implementation behind the same WindowedForecaster interface used by every
other Phase-3 model. Multi-output regression — one forward pass yields
``(N, horizon)`` predictions in one shot.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import PatchTSTConfig, PatchTSTForPrediction

from .base import WindowedForecaster, FitReport

logger = logging.getLogger(__name__)


class PatchTSTForecaster(WindowedForecaster):
    name = "patchtst"
    is_stochastic = True
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.input_chunk_length = int(hp.pop("input_chunk_length", 168))
        self.patch_length = int(hp.pop("patch_length", 16))
        self.patch_stride = int(hp.pop("stride", hp.pop("patch_stride", 8)))
        self.d_model = int(hp.pop("d_model", 64))
        self.num_attention_heads = int(hp.pop("num_attention_heads", 4))
        self.num_hidden_layers = int(hp.pop("num_hidden_layers", 3))
        self.ffn_dim = int(hp.pop("ffn_dim", max(64, self.d_model * 4)))
        self.dropout = float(hp.pop("dropout", 0.1))
        self.batch_size = int(hp.pop("batch_size", 64))
        self.n_epochs = int(hp.pop("n_epochs", 50))
        self.lr = float(hp.pop("lr", 1e-3))
        self.patience = int(hp.pop("patience", 8))
        self._extra = hp
        self._device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        self.model: PatchTSTForPrediction | None = None

    def _build(self) -> PatchTSTForPrediction:
        torch.manual_seed(self.seed)
        cfg = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.input_chunk_length,
            prediction_length=self.horizon,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            attention_dropout=self.dropout,
            num_targets=1,
            scaling="std",
            loss="mse",
        )
        return PatchTSTForPrediction(cfg).to(self._device)

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        # build tensors (B, L, 1), (B, h, 1)
        def to_t(a):
            t = torch.from_numpy(np.asarray(a, dtype=np.float32))
            return t.unsqueeze(-1)
        X_t = to_t(X); y_t = to_t(y)
        train_loader = DataLoader(TensorDataset(X_t, y_t),
                                  batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if X_val is not None:
            X_v = to_t(X_val); y_v = to_t(y_val)
            val_loader = DataLoader(TensorDataset(X_v, y_v),
                                    batch_size=self.batch_size)
        model = self._build()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)
        best_val = float("inf")
        patience_left = self.patience
        best_state = None
        for epoch in range(self.n_epochs):
            model.train()
            train_loss = 0.0
            n = 0
            for xb, yb in train_loader:
                xb = xb.to(self._device); yb = yb.to(self._device)
                opt.zero_grad()
                out = model(past_values=xb, future_values=yb)
                loss = out.loss
                loss.backward()
                opt.step()
                train_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            train_loss /= max(n, 1)

            val_loss = float("nan")
            if val_loader is not None:
                model.eval()
                vl = 0.0; vn = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self._device); yb = yb.to(self._device)
                        out = model(past_values=xb, future_values=yb)
                        vl += float(out.loss.item()) * xb.size(0)
                        vn += xb.size(0)
                val_loss = vl / max(vn, 1)

            if val_loader is None:
                continue
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_left = self.patience
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("[patchtst] early stop @epoch=%d val_loss=%.6f", epoch, best_val)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model

    @torch.no_grad()
    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PatchTST predict before fit")
        self.model.eval()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(-1).to(self._device)
        preds = []
        # batched forward pass
        for start in range(0, X_t.size(0), self.batch_size * 4):
            chunk = X_t[start: start + self.batch_size * 4]
            out = self.model.generate(past_values=chunk)
            # out.sequences shape: (B, num_samples, h, n_channels)
            seq = out.sequences
            mean_pred = seq.mean(dim=1)   # (B, h, n_channels)
            preds.append(mean_pred.squeeze(-1).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)

    def _n_parameters(self) -> int | None:
        if self.model is None:
            return None
        return int(sum(p.numel() for p in self.model.parameters()))
