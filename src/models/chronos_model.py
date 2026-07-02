"""Chronos foundation-model forecaster (zero-shot + LoRA fine-tune).

Zero-shot mode wraps Amazon's ``chronos-forecasting`` pipeline directly.
Fine-tune mode applies PEFT LoRA adapters to the underlying T5 backbone
so that we can adapt the model on each dataset's training split with a
GPU memory budget that fits an RTX 3060 (~12 GB).

Both modes share the same prediction code path — at predict time we run
``ChronosPipeline.predict(context, prediction_length=h, num_samples=...)``
on each test window's lookback and take the median across samples as the
point forecast (this is the standard Chronos point-forecast convention).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .base import WindowedForecaster, FitReport

logger = logging.getLogger(__name__)


class ChronosForecaster(WindowedForecaster):
    """Generic Chronos wrapper.

    Set ``hparams['finetune']=True`` to apply LoRA fine-tuning to the
    underlying T5 backbone (requires GPU); otherwise runs pure zero-shot.
    """

    name = "chronos"
    is_stochastic = False                # zero-shot deterministic; fine-tune seeded
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.pretrained = hp.pop("pretrained", "amazon/chronos-t5-small")
        self.finetune = bool(hp.pop("finetune", False))
        self.finetune_epochs = int(hp.pop("finetune_epochs", 3))
        self.finetune_lr = float(hp.pop("finetune_lr", 1e-4))
        self.lora_r = int(hp.pop("lora_r", 8))
        self.num_samples = int(hp.pop("num_samples", 20))
        self.batch_size_predict = int(hp.pop("batch_size_predict", 16))
        self._extra = hp
        self._device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.pipeline = None             # ChronosPipeline (loaded lazily)

    # ---- variant-aware name (so results files distinguish zero-shot vs FT) ----

    @property
    def variant_name(self) -> str:
        return "chronos_ft" if self.finetune else "chronos_zs"

    def _load_pipeline(self):
        from chronos import ChronosPipeline
        if self.pipeline is not None:
            return
        logger.info("[chronos] loading pretrained=%s on %s", self.pretrained, self._device)
        # chronos 2.x renamed torch_dtype -> dtype
        self.pipeline = ChronosPipeline.from_pretrained(
            self.pretrained,
            device_map=self._device,
            dtype=torch.float32 if self._device == "cpu" else torch.bfloat16,
        )

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        self._load_pipeline()
        if not self.finetune:
            logger.info("[chronos] zero-shot: skipping fit().")
            return
        if self._device == "cpu":
            logger.warning("[chronos] fine-tuning on CPU is impractical; falling back to zero-shot.")
            self.finetune = False
            return
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            logger.error("[chronos] peft missing — falling back to zero-shot.")
            self.finetune = False
            return

        backbone = self.pipeline.model.model    # underlying T5
        target_modules = ["q", "v"]
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=2 * self.lora_r,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        try:
            self.pipeline.model.model = get_peft_model(backbone, lora_cfg).to(self._device)
        except Exception as e:
            logger.warning("[chronos] LoRA wrap failed (%s); skipping fine-tune.", e)
            self.finetune = False
            return

        # Build (context, target) pairs from training windows directly using the
        # Chronos tokenizer + T5 cross-entropy training step. We do a tiny
        # number of epochs to adapt rather than full pre-training.
        self._lora_finetune_loop(X, y)

    def _lora_finetune_loop(self, X: np.ndarray, y: np.ndarray) -> None:
        """Minimal LoRA fine-tuning loop using the Chronos tokenizer.

        Chronos T5 outputs ``tokenizer.config.prediction_length`` tokens
        (64 for all current public checkpoints). When our horizon ``h`` is
        smaller we pad the target on the right with the last observed value
        and mask those positions out of the loss.
        """
        from torch.utils.data import DataLoader, TensorDataset
        tokenizer = self.pipeline.tokenizer
        native_h = int(tokenizer.config.prediction_length)
        if y.shape[1] > native_h:
            logger.warning(
                "[chronos.ft] horizon %d > native pred length %d; truncating during FT.",
                y.shape[1], native_h,
            )
        h_eff = min(y.shape[1], native_h)
        model_inner = self.pipeline.model.model.to(self._device)
        model_inner.train()
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_inner.parameters()),
                                lr=self.finetune_lr)
        # subsample for tractability — LoRA fine-tuning is gradient-light
        idxs = np.random.default_rng(self.seed).choice(
            X.shape[0], size=min(X.shape[0], 512), replace=False
        )
        Xs = torch.from_numpy(X[idxs].astype(np.float32))
        ys = torch.from_numpy(y[idxs].astype(np.float32))
        loader = DataLoader(TensorDataset(Xs, ys), batch_size=4, shuffle=True)
        t0 = time.perf_counter()
        for epoch in range(self.finetune_epochs):
            ep_loss = 0.0; n = 0
            for ctx, tgt in loader:
                ctx = ctx.to(self._device)
                tgt = tgt.to(self._device)
                # pad target on right to native horizon
                B = tgt.size(0)
                if tgt.size(1) < native_h:
                    pad = tgt[:, -1:].repeat(1, native_h - tgt.size(1))
                    tgt_pad = torch.cat([tgt[:, :h_eff], pad], dim=1)
                else:
                    tgt_pad = tgt[:, :native_h]
                try:
                    # tokenizer buckets live on CPU — move inputs there first
                    ctx_cpu = ctx.detach().cpu()
                    tgt_cpu = tgt_pad.detach().cpu()
                    tokens, attention_mask, scale = tokenizer.context_input_transform(ctx_cpu)
                    labels, _ = tokenizer.label_input_transform(tgt_cpu, scale)
                    labels[:, h_eff:] = -100
                    out = model_inner(
                        input_ids=tokens.to(self._device),
                        attention_mask=attention_mask.to(self._device),
                        labels=labels.to(self._device),
                    )
                    loss = out.loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_loss += float(loss.item()) * B
                    n += B
                except Exception as e:
                    logger.warning("[chronos.ft] batch failed: %s: %s", type(e).__name__, e)
                    continue
            logger.info("[chronos.ft] epoch %d loss=%.6f n=%d", epoch + 1,
                        ep_loss / max(n, 1), n)
        elapsed = time.perf_counter() - t0
        logger.info("[chronos.ft] LoRA fine-tune done in %.1fs", elapsed)
        model_inner.eval()

    @torch.no_grad()
    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            self._load_pipeline()
        contexts = [torch.tensor(x, dtype=torch.float32) for x in X]
        preds = np.zeros((X.shape[0], self.horizon), dtype=np.float32)
        bs = max(1, self.batch_size_predict)
        for start in range(0, len(contexts), bs):
            chunk = contexts[start: start + bs]
            try:
                # chronos 2.x: positional `inputs` (list of 1-D tensors), keyword `prediction_length`
                samples = self.pipeline.predict(
                    chunk,
                    prediction_length=self.horizon,
                    num_samples=self.num_samples,
                    limit_prediction_length=False,
                )
                # samples shape: (B, num_samples, h) tensor
                if isinstance(samples, list):
                    arr = torch.stack(samples, dim=0)
                else:
                    arr = samples
                arr = arr.detach().float().cpu().numpy()
                if arr.ndim == 2:           # (B, h) — already point
                    median = arr
                else:
                    median = np.median(arr, axis=1)
                preds[start: start + median.shape[0]] = median.astype(np.float32)
            except Exception as e:
                logger.warning("[chronos] predict chunk failed (%s); using last-value fallback.", e)
                for j, ctx in enumerate(chunk):
                    preds[start + j] = float(ctx[-1])
        return preds

    def _n_parameters(self) -> int | None:
        if self.pipeline is None:
            return None
        try:
            return int(sum(p.numel() for p in self.pipeline.model.parameters()))
        except Exception:
            return None
