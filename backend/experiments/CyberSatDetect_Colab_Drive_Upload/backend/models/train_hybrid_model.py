import os
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
APP_DIR = BACKEND / "app"
DATA_DIR = ROOT / "data" / "reduced"
SPLIT_FILE = BACKEND / "config" / "data_split.json"
MODEL_OUT = APP_DIR / "final_model.keras"
THRESH_OUT = APP_DIR / "thresholds.json"
CHECKPOINT_MODEL = APP_DIR / "checkpoint_model.keras"
CHECKPOINT_INFO = APP_DIR / "checkpoint_info.json"

EPOCHS_PER_CHUNK = 1
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

MAX_CHUNKS = 667
EARLY_STOP_PATIENCE = 10
VAL_CHECK_INTERVAL = 20

W_RECON = 1.0
W_PRED  = 2.0
W_GRAD  = 2.0
W_SEP   = 0.5
ANOMALY_MARGIN = 0.15


def load_npy(path):
    x = np.load(path).astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    return x


def build_model(T, C):
    """
    بناء النموذج الهجين LSTM-GRU حسب المواصفات
    
    العمارة:
    - Encoder: Conv1D(2x) + LSTM + GRU
    - Latent Bridge: RepeatVector
    - Decoder: GRU + LSTM + TimeDistributed(Dense)
    - Prediction Head: للتنبؤ بخطوة واحدة
    """
    
    # ================== ENCODER ==================
    inp = layers.Input(shape=(T, C), name="input")
    
    # Conv1D layers with causal padding
    x = layers.Conv1D(64, 5, padding="causal", activation="relu", name="conv1d_1")(inp)
    x = layers.LayerNormalization(name="layernorm_1")(x)
    
    x = layers.Conv1D(64, 3, padding="causal", activation="relu", name="conv1d_2")(x)
    x = layers.LayerNormalization(name="layernorm_2")(x)
    
    # LSTM for long-term dependencies
    x = layers.LSTM(128, return_sequences=True, name="lstm_enc")(x)
    x = layers.Dropout(0.1, name="dropout_lstm")(x)
    
    # GRU for short-term patterns (Hybrid!)
    x = layers.GRU(64, return_sequences=False, name="gru_enc")(x)
    
    # ================== LATENT BRIDGE ==================
    x_latent = x
    x = layers.RepeatVector(T, name="repeat_vector")(x)
    
    # ================== DECODER ==================
    # GRU for reconstruction
    d = layers.GRU(64, return_sequences=True, name="gru_dec")(x)
    d = layers.Dropout(0.1, name="dropout_gru")(d)
    
    # LSTM for temporal coherence
    d = layers.LSTM(128, return_sequences=True, name="lstm_dec")(d)
    
    # TimeDistributed Dense for point-wise reconstruction
    recon = layers.TimeDistributed(
        layers.Dense(C, name="td_dense_recon"),
        name="reconstruction"
    )(d)
    
    # ================== PREDICTION HEAD ==================
    # From latent, predict next timestep
    pred_feat = layers.Dense(64, activation="relu", name="pred_fc1")(x_latent)
    pred = layers.Dense(C, name="prediction")(pred_feat)
    
    # ================== BUILD MODEL ==================
    model = Model(inputs=inp, outputs=[recon, pred], name="Hybrid_LSTM_GRU_AE")
    
    return model


def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))

def grad_mse(x_true, x_recon):
    dx_true  = x_true[:, 1:, :] - x_true[:, :-1, :]
    dx_recon = x_recon[:, 1:, :] - x_recon[:, :-1, :]
    return tf.reduce_mean(tf.square(dx_true - dx_recon))


def build_pseudo_anomalies(x, seed=None):
    """
    بناء anomalies زائفة للـ self-supervised learning
    
    تستخدم نوعين من الهجمات:
    1. Freeze Attack: تثبيت القيمة
    2. Pattern Shift Attack: إزاحة دورانية
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    batch_size = tf.shape(x)[0]
    
    # ============ Freeze Attack ============
    freeze = freeze_attack_tf(x)
    
    # ============ Pattern Shift Attack ============
    pattern = pattern_shift_attack_tf(x, shift=20)
    
    # ============ Mix Both ============
    selector = tf.random.uniform((batch_size, 1, 1), minval=0.0, maxval=1.0)
    pseudo = tf.where(selector < 0.5, freeze, pattern)
    
    return pseudo


def compute_sep_loss_tf(scores_normal, scores_pseudo, margin=0.15):
    """
    حساب Separation Loss
    
    L_sep = (1/n) Σ max(0, δ - (s_pseudo - s_normal))
    
    حيث:
    - δ = anomaly margin (0.15)
    - s_pseudo = درجات الـ pseudo-anomalies
    - s_normal = درجات البيانات الطبيعية
    """
    # نريد scores_pseudo > scores_normal + margin
    # بمعنى: الفرق يجب أن يكون أكبر من الـ margin
    
    diff = scores_pseudo - scores_normal
    violation = tf.maximum(0.0, margin - diff)
    loss = tf.reduce_mean(violation)
    
    return loss


def freeze_attack_tf(x):
    baseline = x[:, :1, :]
    frozen = tf.repeat(baseline, repeats=tf.shape(x)[1], axis=1)
    return frozen + 0.8


def pattern_shift_attack_tf(x, shift=15):
    shifted = tf.roll(x, shift=shift, axis=1)
    return (-1.2 * shifted) + 0.6


def build_pseudo_anomalies(x):
    freeze = freeze_attack_tf(x)
    pattern = pattern_shift_attack_tf(x)
    selector = tf.random.uniform((tf.shape(x)[0], 1, 1), minval=0.0, maxval=1.0)
    return tf.where(selector < 0.5, freeze, pattern)


def batch_scores_tf(x_true, x_recon, x_pred):
    e_recon = tf.reduce_mean(tf.square(x_true - x_recon), axis=[1, 2])
    # x_pred يتنبأ بـ timestep واحد فقط (الخطوة التالية)
    e_pred = tf.reduce_mean(tf.square(x_true[:, -1:, :] - x_pred), axis=[1, 2])

    dx_true = x_true[:, 1:, :] - x_true[:, :-1, :]
    dx_recon = x_recon[:, 1:, :] - x_recon[:, :-1, :]
    e_grad = tf.reduce_mean(tf.square(dx_true - dx_recon), axis=[1, 2])

    return (W_RECON * e_recon) + (W_PRED * e_pred) + (W_GRAD * e_grad)


def compute_total_loss_with_sep(model, X_batch, margin=0.15):
    """
    تحسب الخسارة الكلية مع L_sep
    
    L_total = W_recon·L_recon + W_pred·L_pred + W_grad·L_grad + W_sep·L_sep
    """
    with tf.GradientTape() as tape:
        # Forward pass على البيانات الطبيعية
        recon, pred = model(X_batch, training=True)
        
        # حساب الخسائر المختلفة
        e_recon = tf.reduce_mean(tf.square(X_batch - recon))
        e_pred = tf.reduce_mean(tf.square(X_batch[:, 1:, :] - pred))
        
        dx_true = X_batch[:, 1:, :] - X_batch[:, :-1, :]
        dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
        e_grad = tf.reduce_mean(tf.square(dx_true - dx_recon))
        
        # حساب الدرجات للبيانات الطبيعية
        scores_normal = batch_scores_tf(X_batch, recon, pred)
        
        # بناء pseudo-anomalies
        X_pseudo = build_pseudo_anomalies(X_batch)
        
        # Forward pass على pseudo-anomalies
        recon_pseudo, pred_pseudo = model(X_pseudo, training=True)
        
        # حساب الدرجات للـ pseudo-anomalies
        scores_pseudo = batch_scores_tf(X_pseudo, recon_pseudo, pred_pseudo)
        
        # حساب Separation Loss
        e_sep = compute_sep_loss_tf(scores_normal, scores_pseudo, margin=margin)
        
        # المجموع
        loss = (W_RECON * e_recon + 
                W_PRED * e_pred + 
                W_GRAD * e_grad + 
                W_SEP * e_sep)
    
    return loss, tape


def compute_scores(model, X):
    recon, pred = model.predict(X, verbose=0)

    e_recon = np.mean((X - recon)**2, axis=(1,2))

    pred_reshaped = np.reshape(pred, (-1, 1, 1))
    e_pred = np.mean((X[:, -1:, :] - pred_reshaped)**2, axis=(1,2))

    dx_true  = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon)**2, axis=(1,2))

    return (W_RECON*e_recon) + (W_PRED*e_pred) + (W_GRAD*e_grad)


def save_threshold(scores):
    mean = float(np.mean(scores))
    std  = float(np.std(scores))

    data = {
        "mean": mean,
        "std": std,
        "thresholds": {
            "3sigma": mean + 3*std,
            "p99": float(np.quantile(scores, 0.99)),
            "p995": float(np.quantile(scores, 0.995)),
            "p997": float(np.quantile(scores, 0.997)),
        },
        "weights": {
            "W_RECON": W_RECON,
            "W_PRED": W_PRED,
            "W_GRAD": W_GRAD,
            "W_SEP": W_SEP,
        },
        "self_supervised": {
            "pseudo_anomalies": ["freeze", "pattern_shift"],
            "anomaly_margin": ANOMALY_MARGIN,
        }
    }

    with open(THRESH_OUT, "w") as f:
        json.dump(data, f, indent=4)


def save_checkpoint(model, last_chunk, best_val_loss, best_chunk, no_improve_count, current_lr):
    """حفظ checkpoint مع معلومات التقدم"""
    # حفظ النموذج
    model.save(CHECKPOINT_MODEL)
    
    # حفظ معلومات التقدم
    checkpoint_info = {
        "last_chunk": int(last_chunk),
        "best_val_loss": float(best_val_loss),
        "best_chunk": int(best_chunk),
        "no_improve_count": int(no_improve_count),
        "current_lr": float(current_lr)
    }
    
    with open(CHECKPOINT_INFO, "w") as f:
        json.dump(checkpoint_info, f, indent=4)


def load_checkpoint():
    """حمل checkpoint إن وجد، وإرجاع معلومات التقدم
    
    Returns:
        (model, start_chunk, checkpoint_info) أو (None, 0, None) إذا لم يوجد checkpoint
    """
    if not CHECKPOINT_MODEL.exists() or not CHECKPOINT_INFO.exists():
        return None, 0, None
    
    try:
        # حمل معلومات التقدم
        with open(CHECKPOINT_INFO, "r") as f:
            checkpoint_info = json.load(f)
        
        # حمل النموذج
        model = tf.keras.models.load_model(CHECKPOINT_MODEL, custom_objects={'tf': tf})
        
        start_chunk = checkpoint_info["last_chunk"] + 1
        
        print(f"\n{'='*70}")
        print(f"Resuming training from checkpoint!")
        print(f"Last completed chunk: {checkpoint_info['last_chunk']}")
        print(f"Starting from chunk: {start_chunk}")
        print(f"Best val_loss so far: {checkpoint_info['best_val_loss']:.6f}")
        print(f"{'='*70}\n")
        
        return model, start_chunk, checkpoint_info
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, 0, None


def main():

    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)

    train_files = [DATA_DIR / name for name in split["train"]]
    val_files = [DATA_DIR / name for name in split["validation"]]

    print("Train files:", len(train_files))
    print("Validation files:", len(val_files))

    # ===== Load Checkpoint if exists =====
    model, start_chunk, checkpoint_info = load_checkpoint()
    
    if checkpoint_info is not None:
        # استكمال من checkpoint
        best_val_loss = checkpoint_info["best_val_loss"]
        no_improve_count = checkpoint_info["no_improve_count"]
        best_chunk = checkpoint_info["best_chunk"]
        current_lr = checkpoint_info["current_lr"]
    else:
        # بداية جديدة
        print(f"\n{'='*70}")
        print(f"Starting training with early stopping and LR scheduling")
        print(f"{'='*70}\n")
        
        X0 = load_npy(train_files[0])
        _, T, C = X0.shape
        model = build_model(T, C)
        
        best_val_loss = float('inf')
        no_improve_count = 0
        best_chunk = 0
        current_lr = LEARNING_RATE
        start_chunk = 0
    
    optimizer = tf.keras.optimizers.Adam(current_lr)
    all_train_losses = []
    all_val_losses = []

    # ===== Training Loop =====
    for chunk_idx in range(start_chunk, min(start_chunk + MAX_CHUNKS - start_chunk, len(train_files))):
        fp = train_files[chunk_idx]
        X = load_npy(fp)
        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.shuffle(20000).batch(BATCH_SIZE)

        chunk_losses = []

        for epoch in range(EPOCHS_PER_CHUNK):
            for batch in ds:
                with tf.GradientTape() as tape:
                    recon, pred = model(batch, training=True)
                    pseudo_batch = build_pseudo_anomalies(batch)
                    pseudo_recon, pseudo_pred = model(pseudo_batch, training=True)

                    loss_recon = mse(batch, recon)
                    pred_reshaped = tf.reshape(pred, [-1, 1, 1])
                    loss_pred  = mse(batch[:, -1:, :], pred_reshaped)
                    loss_grad  = grad_mse(batch, recon)
                    normal_scores = batch_scores_tf(batch, recon, pred)
                    pseudo_scores = batch_scores_tf(pseudo_batch, pseudo_recon, pseudo_pred)
                    loss_sep = tf.reduce_mean(
                        tf.nn.relu(ANOMALY_MARGIN - (pseudo_scores - normal_scores))
                    )

                    total_loss = (W_RECON*loss_recon) + \
                                 (W_PRED*loss_pred) + \
                                 (W_GRAD*loss_grad) + \
                                 (W_SEP*loss_sep)

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                chunk_losses.append(float(total_loss))

        avg_train_loss = np.mean(chunk_losses)
        all_train_losses.append(avg_train_loss)

        # ===== Validation Check Every 20 Chunks =====
        if (chunk_idx + 1) % VAL_CHECK_INTERVAL == 0 or chunk_idx == 0:
            val_files_sample = random.sample(val_files, min(30, len(val_files)))
            val_scores_list = []

            for val_fp in val_files_sample:
                X_val = load_npy(val_fp)
                val_scores = compute_scores(model, X_val)
                val_scores_list.extend(val_scores)

            val_loss = np.mean(val_scores_list)
            all_val_losses.append(val_loss)

            print(f"[Chunk {chunk_idx+1:3d}/667] "
                  f"train_loss: {avg_train_loss:.6f} | "
                  f"val_loss: {val_loss:.6f} | "
                  f"lr: {current_lr:.6f}")

            # ===== Early Stopping Check =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_chunk = chunk_idx + 1
                no_improve_count = 0
                model.save(APP_DIR / "best_model.keras")
                print(f"  → Best validation loss! Saved best_model.keras")
            else:
                no_improve_count += 1
                print(f"  → No improvement ({no_improve_count}/{EARLY_STOP_PATIENCE})")

            # ===== Learning Rate Schedule =====
            if no_improve_count >= 5 and no_improve_count % 5 == 0:
                current_lr = max(current_lr * 0.5, 1e-5)
                optimizer = tf.keras.optimizers.Adam(current_lr)
                print(f"  → Learning rate reduced to: {current_lr:.6f}")

            # ===== Save Checkpoint Every 20 Chunks =====
            save_checkpoint(model, chunk_idx, best_val_loss, best_chunk, no_improve_count, current_lr)
            print(f"  → Saved checkpoint at chunk {chunk_idx}")


            # ===== Early Stopping =====
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n{'='*70}")
                print(f"Early stopping at chunk {chunk_idx+1}!")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"{'='*70}\n")
                break

    # ===== Load Best Model =====
    print("Loading best model...")
    best_model_path = APP_DIR / "best_model.keras"
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path, 
                                          custom_objects={'tf': tf})
        print(f"Best model loaded from chunk {best_chunk}")
    else:
        print("No best model saved, using final model")

    # ===== Final Threshold Calculation =====
    print("\nCalculating thresholds from all validation files...")
    all_scores = []
    for fp in val_files:
        X = load_npy(fp)
        s = compute_scores(model, X)
        all_scores.extend(s)

    all_scores = np.array(all_scores)
    save_threshold(all_scores)

    model.save(MODEL_OUT)
    print("\nTraining complete!")
    print(f"Best val_loss: {best_val_loss:.6f} at chunk {best_chunk}")
    print(f"Model saved to: {MODEL_OUT}")
    print(f"Thresholds saved to: {THRESH_OUT}")

if __name__ == "__main__":
    main()