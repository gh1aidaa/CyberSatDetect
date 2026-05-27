"""
Experimental Hybrid LSTM-GRU autoencoder with augmented inputs (100, C).

Not used by production inference paths.
"""

from __future__ import annotations

from typing import Literal

import tensorflow as tf
from tensorflow.keras import Model, layers

ReconTarget = Literal["original_only", "all_features"]


def pattern_order_head(latent: tf.Tensor, name_prefix: str = "pattern_order") -> tf.Tensor:
    o = layers.Dense(32, activation="relu", name=f"{name_prefix}_dense1")(latent)
    o = layers.Dropout(0.1, name=f"{name_prefix}_drop")(o)
    return layers.Dense(1, activation="sigmoid", name=f"{name_prefix}_sigmoid")(o)


def build_univariate_pattern_model(
    T: int = 100,
    C_in: int = 8,
    recon_target: ReconTarget = "original_only",
) -> Model:
    recon_dim = 1 if recon_target == "original_only" else int(C_in)

    inp = layers.Input(shape=(T, C_in), name="input_aug")

    x = layers.Conv1D(64, 5, padding="causal", activation="relu", name="enc_conv1")(inp)
    x = layers.LayerNormalization(name="enc_ln1")(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu", name="enc_conv2")(x)
    x = layers.LayerNormalization(name="enc_ln2")(x)
    x = layers.LSTM(128, return_sequences=True, name="enc_lstm")(x)
    x = layers.Dropout(0.1, name="enc_drop")(x)
    latent = layers.GRU(64, return_sequences=False, name="enc_gru")(x)

    rv = layers.RepeatVector(T, name="repeat")(latent)
    d = layers.GRU(64, return_sequences=True, name="dec_gru")(rv)
    d = layers.Dropout(0.1, name="dec_drop")(d)
    d = layers.LSTM(128, return_sequences=True, name="dec_lstm")(d)
    recon = layers.TimeDistributed(layers.Dense(recon_dim), name="reconstruction")(d)

    p = layers.Dense(64, activation="relu", name="pred_fc1")(latent)
    pred10 = layers.Dense(10, name="pred_dense10")(p)
    pred10 = layers.Reshape((10, 1), name="pred_last10")(pred10)

    order = pattern_order_head(latent, name_prefix="pattern_order")

    return Model(
        inputs=inp,
        outputs=[recon, pred10, order],
        name=f"UnivariatePatternHybrid_{recon_target}",
    )
