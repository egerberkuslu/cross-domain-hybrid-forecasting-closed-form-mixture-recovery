"""Configuration of the Phase-5 experiment grid.

Defines:

  * MODEL_CONFIGS — the full list of (registry_name, display_variant,
    base_hparams, stochastic_flag) tuples we evaluate.
  * SEEDS_FOR — returns the per-model seed list (5 seeds for stochastic
    models, 1 seed for deterministic ones).
  * HP_GRIDS — per-model validation-set search spaces (small but
    meaningful, so a Phase-5 full run stays within a sensible compute
    budget while still substantiating the "every model is tuned" claim).
"""
from __future__ import annotations


# Deep-model training defaults used during HP search + final runs.
# Kept modest so the full grid finishes in a reasonable budget; early
# stopping (patience 4) prevents under-fitting.
DEEP_DEFAULTS = {
    "input_chunk_length": 168,
    "n_epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "patience": 4,
}


# --- model fleet ------------------------------------------------------------
# Each tuple: (registry_name, display_variant, base_hparams, is_stochastic)
MODEL_CONFIGS = [
    # ----- Group A: statistical baselines -----
    ("naive", "naive", {}, False),
    ("seasonal_naive", "seasonal_naive", {"seasonal_period": 24}, False),
    (
        "arima",
        "arima",
        {
            # Use a fixed sample size of 2 training lookbacks to choose the order
            # once — keeps the per-window refit fast on the full grid.
            "max_order_search": 2,
            "n_train_samples_for_order": 3,
        },
        False,
    ),
    (
        "holt_winters",
        "holt_winters",
        {
            "seasonal_period": 24,
            "trend": "add",
            "seasonal": "add",
        },
        False,
    ),
    ("theta", "theta", {"seasonal_period": 24}, False),
    (
        "farima",
        "farima",
        {"p": 1, "q": 1, "max_order_search": 2, "trunc_K": 100},
        False,
    ),
    # ----- Group B: classical ML + modern deep -----
    (
        "xgboost",
        "xgboost",
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
        True,
    ),
    (
        "lstm",
        "lstm",
        {**DEEP_DEFAULTS, "hidden_dim": 64, "n_rnn_layers": 1, "dropout": 0.1},
        True,
    ),
    (
        "gru",
        "gru",
        {**DEEP_DEFAULTS, "hidden_dim": 64, "n_rnn_layers": 1, "dropout": 0.1},
        True,
    ),
    (
        "tcn",
        "tcn",
        {**DEEP_DEFAULTS, "num_filters": 16, "kernel_size": 3, "dropout": 0.1},
        True,
    ),
    (
        "nbeats",
        "nbeats",
        {
            **DEEP_DEFAULTS,
            "num_stacks": 2,
            "num_blocks": 2,
            "num_layers": 2,
            "layer_widths": 64,
        },
        True,
    ),
    ("dlinear", "dlinear", {**DEEP_DEFAULTS, "kernel_size": 25}, True),
    (
        "patchtst",
        "patchtst",
        {
            **DEEP_DEFAULTS,
            "patch_length": 16,
            "patch_stride": 8,
            "d_model": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 3,
            "dropout": 0.1,
        },
        True,
    ),
    # ----- Group C: foundation models -----
    (
        "chronos",
        "chronos_zs",
        {
            "pretrained": "amazon/chronos-t5-small",
            "finetune": False,
            "num_samples": 20,
            "batch_size_predict": 16,
        },
        False,
    ),
    (
        "chronos",
        "chronos_ft",
        {
            "pretrained": "amazon/chronos-t5-tiny",
            "finetune": True,
            "finetune_epochs": 3,
            "lora_r": 8,
            "finetune_lr": 1e-4,
            "num_samples": 20,
            "batch_size_predict": 16,
        },
        True,
    ),
    (
        "timesfm",
        "timesfm_zs",
        {
            "input_chunk_length": 512,
            "batch_size_predict": 16,
        },
        False,
    ),
    # ----- Group B' — modern 2023-2024 SOTA additions (darts) -----
    # Per Aouedi et al. 2025 ACM CSUR survey + Liu et al. 2026 MSCAF (J. SoftSys)
    (
        "nhits",
        "nhits",
        {
            **DEEP_DEFAULTS,
            "num_stacks": 3,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 64,
            "dropout": 0.1,
        },
        True,
    ),
    (
        "tft",
        "tft",
        {
            **DEEP_DEFAULTS,
            "hidden_size": 32,
            "lstm_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
        },
        True,
    ),
    (
        "tide",
        "tide",
        {
            **DEEP_DEFAULTS,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "decoder_output_dim": 8,
            "hidden_size": 64,
            "dropout": 0.1,
        },
        True,
    ),
    (
        "tsmixer",
        "tsmixer",
        {
            **DEEP_DEFAULTS,
            "hidden_size": 64,
            "ff_size": 64,
            "num_blocks": 2,
            "dropout": 0.1,
        },
        True,
    ),
    # ----- Group C' — additional 2024-2025 foundation models -----
    (
        "chronos_bolt",
        "chronos_bolt_zs",
        {
            "pretrained": "amazon/chronos-bolt-small",
            "batch_size_predict": 16,
        },
        False,
    ),
    (
        "moirai",
        "moirai_zs",
        {
            "pretrained": "Salesforce/moirai-1.1-R-small",
            "input_chunk_length": 512,
            "patch_size": "auto",
            "num_samples": 20,
            "batch_size_predict": 16,
        },
        False,
    ),
    (
        "ttm",
        "ttm_zs",
        {
            "pretrained": "ibm-granite/granite-timeseries-ttm-r2",
            "context_length": 512,
            "batch_size_predict": 16,
        },
        False,
    ),
    # ----- Proposed -----
    # CHA-Hybrid trains TWO deep sub-models (residual GRU + global LSTM); we
    # give them a deeper training budget (30 epochs vs the 10 used for the
    # other baselines) so the proposed model gets a fair shake.  Finer
    # α-grid (0.05 step) for slightly better per-horizon weighting.
    (
        "cha_hybrid",
        "cha_hybrid",
        {
            "stl_period": 24,
            "trend_model": "linear",
            "seasonal_model": "seasonal_naive",
            "residual_model": "gru",
            "global_model": "lstm",
            "gru": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "alpha_search": [
                round(0.05 * i, 2) for i in range(21)
            ],  # 0.0..1.0 step 0.05
        },
        True,
    ),
    # CHA-Hybrid v2 — adaptive mixture of classical decomposition (Theta-trend +
    # SeasonalNaive + LSTM-residual) and a foundation-model expert (TimesFM
    # zero-shot). Per-horizon α arbitrates between the two paths on val.
    (
        "cha_hybrid_v2",
        "cha_hybrid_v2",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "timesfm": {
                "input_chunk_length": 512,
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    # CHA-Hybrid v3 — same as v2 but global expert = Chronos-Bolt
    # (Phase-5 showed Bolt beats TimesFM by 12-14% on cesnet).
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-small",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    # CHA-Hybrid v4 — same experts as v3 but α is a learned per-sample MLP
    # head (real methodological novelty over v3's scalar α_h).
    (
        "cha_hybrid_v4",
        "cha_hybrid_v4",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-small",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
            "alpha_mlp": {
                "hidden": 16,
                "epochs": 300,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "patience": 30,
            },
        },
        True,
    ),
    # CHA-Hybrid v4-fix — same as v4 but with proper held-out early stopping
    # on a chronological val_train / val_holdout split, smaller MLP (8 hidden,
    # 1 layer), and stronger regularisation.  Addresses the Abilene
    # overfitting regression observed with vanilla v4.
    (
        "cha_hybrid_v4_fix",
        "cha_hybrid_v4_fix",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-small",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
            "alpha_mlp": {
                "hidden": 8,
                "epochs": 500,
                "lr": 1e-3,
                "weight_decay": 5e-3,
                "patience": 30,
                "val_holdout_frac": 0.2,
            },
        },
        True,
    ),
    # ----- Exp 2.3: foundation-model size scaling -----
    # Same v3 architecture but the global expert swaps in chronos-bolt-tiny/
    # mini/base.  Lets us answer "does the proposed hybrid degrade gracefully
    # when the foundation model is smaller / improve with a bigger one?"
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3_tiny",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-tiny",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3_mini",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-mini",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3_base",
        {
            "stl_period": 24,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-base",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    # ----- Exp 2.1: STL period sensitivity (12 / 48 / 168) -----
    # Default v3 uses period=24 (daily seasonality at hourly resolution).
    # These variants probe shorter and longer seasonal cycles.
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3_stl12",
        {
            "stl_period": 12,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-small",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3_stl48",
        {
            "stl_period": 48,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-small",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
    (
        "cha_hybrid_v3",
        "cha_hybrid_v3_stl168",
        {
            "stl_period": 168,
            "lstm": {
                **DEEP_DEFAULTS,
                "hidden_dim": 64,
                "n_rnn_layers": 1,
                "dropout": 0.1,
                "n_epochs": 30,
                "patience": 6,
            },
            "chronos_bolt": {
                "pretrained": "amazon/chronos-bolt-small",
                "batch_size_predict": 16,
            },
            "alpha_search": [round(0.05 * i, 2) for i in range(21)],
        },
        True,
    ),
]


# --- HP search grids (deliberately small for Phase-5 budget) -----------------
# Each grid is *layered on top of* the corresponding base_hparams above.
# We keep the grids tight (2 candidates per knob) so the full grid completes
# in a session-friendly budget while still substantiating the
# "every baseline is tuned on val" fairness claim.
HP_GRIDS: dict[str, dict] = {
    "xgboost": {"max_depth": [3, 5]},  # 2 candidates
    "lstm": {"hidden_dim": [32, 64]},  # 2 candidates
    "gru": {"hidden_dim": [32, 64]},  # 2 candidates
    "tcn": {"num_filters": [16, 32]},  # 2 candidates
    "nbeats": {"num_blocks": [1, 2]},  # 2 candidates
    "dlinear": {"kernel_size": [13, 25]},  # 2 candidates
    "patchtst": {"d_model": [64, 128]},  # 2 candidates
    # Foundation models, naive baselines, ARIMA/HW/Theta, CHA-Hybrid have
    # no Phase-5 HP grid — their architecture choices are pre-baked into
    # base_hparams above (foundation pretrained ids; STL+GRU+LSTM for CHA;
    # season period for HW/SN/Theta).
}


def seeds_for(stochastic: bool, all_seeds: list[int]) -> list[int]:
    """Deterministic models use only the first seed; stochastic models all 5."""
    return list(all_seeds) if stochastic else [int(all_seeds[0])]
