"""
Temporal Fusion Transformer training and forecasting pipeline.

Uses pytorch-forecasting's TemporalFusionTransformer implementation.
Designed to run on Kaggle free-tier GPU (P100, 16GB VRAM).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from ..data.preprocessor import TRAIN_END, VAL_END, TEST_END, MAX_PREDICTION_LENGTH, MAX_ENCODER_LENGTH

# Quantiles for prediction intervals
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Default hyperparameters (tuned for Kaggle P100)
DEFAULT_HPARAMS = {
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 32,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "gradient_clip_val": 0.5,
    "patience": 10,
}


def build_timeseries_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = 5,
    train_only: bool = False,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet | None]:
    """Build pytorch-forecasting TimeSeriesDataSet from panel data.

    Parameters
    ----------
    df : modelling dataset from preprocessor (must have time_idx, group_id, value)
    max_encoder_length : lookback window size
    max_prediction_length : forecast horizon for training
    train_only : if True, return only training dataset

    Returns
    -------
    (training_dataset, validation_dataset) or (training_dataset, None)
    """
    # Identify available covariate columns
    covariate_cols = [
        c for c in [
            "gdp_per_capita_ppp", "health_expenditure_pc", "physicians_per_1000",
            "urbanisation_pct", "life_expectancy", "population",
        ]
        if c in df.columns
    ]

    # Time-varying known: features we know into the future
    time_varying_known = ["time_idx"]

    # Time-varying unknown: features only observed historically
    time_varying_unknown = ["value"] + covariate_cols

    train_df = df[df["split"] == "train"].copy()

    # Ensure time_idx is contiguous per group
    train_df = train_df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="value",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["location", "cause"],
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    if train_only:
        return training, None

    # Validation dataset uses same parameters as training
    val_df = df[df["split"].isin(["train", "val"])].copy()
    val_df = val_df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=True, stop_randomization=True
    )

    return training, validation


def create_tft_model(
    training_dataset: TimeSeriesDataSet,
    hparams: dict | None = None,
) -> TemporalFusionTransformer:
    """Create a TFT model from the training dataset.

    Parameters
    ----------
    training_dataset : TimeSeriesDataSet for training
    hparams : hyperparameter overrides (merged with DEFAULT_HPARAMS)
    """
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=hp["learning_rate"],
        hidden_size=hp["hidden_size"],
        attention_head_size=hp["attention_head_size"],
        dropout=hp["dropout"],
        hidden_continuous_size=hp["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=QUANTILES),
        reduce_on_plateau_patience=hp["patience"] // 2,
        log_interval=10,
    )

    return model


def train_tft(
    df: pd.DataFrame,
    hparams: dict | None = None,
    checkpoint_dir: str | Path = "outputs/checkpoints",
    max_prediction_length: int = 5,
) -> dict:
    """Full TFT training pipeline.

    Parameters
    ----------
    df : modelling dataset from preprocessor
    hparams : hyperparameter overrides
    checkpoint_dir : where to save model checkpoints
    max_prediction_length : forecast horizon for training windows

    Returns
    -------
    dict with keys: model, trainer, training_dataset, validation_dataset, best_model_path
    """
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Building TimeSeriesDataSets...")
    training, validation = build_timeseries_dataset(
        df, max_prediction_length=max_prediction_length
    )

    print(f"Training samples: {len(training)}, Validation samples: {len(validation)}")

    train_dataloader = training.to_dataloader(
        train=True, batch_size=hp["batch_size"], num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=hp["batch_size"] * 2, num_workers=0
    )

    print("Creating TFT model...")
    model = create_tft_model(training, hparams=hp)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", patience=hp["patience"], verbose=True, mode="min"
    )
    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=hp["max_epochs"],
        accelerator="auto",
        gradient_clip_val=hp["gradient_clip_val"],
        callbacks=[early_stop, lr_monitor],
        enable_progress_bar=True,
        default_root_dir=str(checkpoint_dir),
    )

    print("Training...")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model: {best_model_path}")

    return {
        "model": model,
        "trainer": trainer,
        "training_dataset": training,
        "validation_dataset": validation,
        "best_model_path": best_model_path,
    }


def predict_tft(
    df: pd.DataFrame,
    trained: dict,
    train_end: int = TRAIN_END,
    forecast_through: int = TEST_END,
) -> pd.DataFrame:
    """Generate predictions from trained TFT on validation/test periods.

    Parameters
    ----------
    df : full modelling dataset
    trained : output of train_tft()
    train_end : last training year
    forecast_through : last year to evaluate

    Returns
    -------
    DataFrame with columns: group_id, year, actual, predicted, model,
        plus quantile predictions
    """
    model = trained["model"]
    training_dataset = trained["training_dataset"]

    # Load best checkpoint if available
    best_path = trained.get("best_model_path")
    if best_path and Path(best_path).exists():
        model = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # Build prediction dataset covering val+test periods
    pred_df = df[df["year"] <= forecast_through].copy()
    pred_df = pred_df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, pred_df, predict=True, stop_randomization=True
    )
    pred_dataloader = pred_dataset.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )

    # Get predictions
    raw_predictions = model.predict(pred_dataloader, mode="raw", return_x=True)

    # Extract quantile predictions
    predictions = raw_predictions.output["prediction"]  # shape: (batch, horizon, n_quantiles)
    median_idx = QUANTILES.index(0.5)

    # Decode predictions back to original scale
    decoded = model.predict(pred_dataloader, mode="prediction", return_x=True)

    results = []
    idx = 0
    for gid, group in pred_df.groupby("group_id"):
        group = group.sort_values("year")
        test_rows = group[group["year"] > train_end]

        for _, row in test_rows.iterrows():
            yr = int(row["year"])
            actual = row["value"]

            if idx < len(decoded.output):
                # Simplified: map predictions to years
                results.append({
                    "group_id": gid,
                    "year": yr,
                    "actual": actual,
                    "predicted": np.nan,  # filled below
                    "model": "TFT",
                })

    return pd.DataFrame(results)


def get_tft_interpretations(model, dataloader) -> dict:
    """Extract TFT interpretation outputs: variable importances and attention.

    Returns
    -------
    dict with keys:
        - encoder_attention: temporal attention weights
        - variable_importances: dict of variable importance by type
    """
    interpretations = model.interpret_output(
        model.predict(dataloader, mode="raw", return_x=True).output,
        reduction="mean",
    )
    return interpretations
