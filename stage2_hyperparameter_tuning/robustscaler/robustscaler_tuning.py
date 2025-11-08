import os
import sys
import json
import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# === SLURM ARRAY SETUP ===
# Get job index from SLURM array (1-indexed)
job_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1

# === Hyperparameter Grid ===
encoding_dims = [64, 128, 256, 1024, 2048]
layer_configs = [
    [4096],
    [4096, 1024],
    [4096, 2048, 1024],
    [4096, 2048, 1024, 512],
    [4096, 2048, 1024, 512, 256]
]

# Generate all combinations
from itertools import product
param_grid = list(product(encoding_dims, layer_configs))

# Get current configuration
encoding_dim, layer_config = param_grid[job_index]

# === TensorFlow CPU Optimization ===
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(2)

# === Disable GPU ===
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === Step 1: Load Data ===
data_path = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/labeled_df_16_feb.csv"
df = pl.read_csv(data_path, infer_schema_length=None)

# === Step 2: Filter Columns ===
non_numeric_cols = [
    "Assembly", "Domain", "Phylum", "Class", "Order", "Family",
    "Genus", "Species", "Genome accessions", "Label"
]
numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes)
                if col not in non_numeric_cols and dtype == pl.Float64]

# === Step 3: Prepare Data ===
X = df.select(numeric_cols).fill_null(0.0).to_numpy().astype(np.float32)
# X = np.clip(X, a_min=0, a_max=1000)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# === Step 4: Build Model ===
input_dim = X_scaled.shape[1]
dropout_rate = 0.2
input_layer = keras.Input(shape=(input_dim,))
x = input_layer

# Encoder
for units in layer_config:
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

# Bottleneck
encoded = layers.Dense(encoding_dim, activation="relu")(x)
x = layers.BatchNormalization()(encoded)
x = layers.Dropout(dropout_rate)(x)

# Decoder (mirror)
for units in reversed(layer_config):
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

# Output layer
decoded = layers.Dense(input_dim)(x)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# === Step 5: Train Model ===
# Record start time
start_time = time.time()

# Custom callback to track training metrics
class TrainingMetricsCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

training_metrics = TrainingMetricsCallback()

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        training_metrics
    ],
    verbose=1
)

# Record total training time
total_training_time = time.time() - start_time

# === Step 6: Evaluate ===
X_pred = autoencoder.predict(X_scaled)

# Base output directory
base_output_dir = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/robustscaler"

# Create directories for different output types
plots_dir = os.path.join(base_output_dir, "plots")
histograms_dir = os.path.join(base_output_dir, "histograms")
summaries_dir = os.path.join(base_output_dir, "summaries")
results_dir = os.path.join(base_output_dir, "results")

# Create directories if they don't exist
for directory in [plots_dir, histograms_dir, summaries_dir, results_dir]:
    os.makedirs(directory, exist_ok=True)

# Basic metrics
mse = mean_squared_error(X_scaled, X_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(X_scaled, X_pred)
r2 = r2_score(X_scaled, X_pred)

# Range-aware MSE
feature_ranges = X_scaled.max(axis=0) - X_scaled.min(axis=0)
range_mse = np.mean(((X_scaled - X_pred) ** 2) / (feature_ranges ** 2 + 1e-8))

# Calculate per-feature range_mse
squared_errors = (X_scaled - X_pred) ** 2
per_feature_squared_errors = np.mean(squared_errors, axis=0)
per_feature_range_mse = per_feature_squared_errors / (feature_ranges ** 2 + 1e-8)

# Baseline MSE (variance-based)
baseline_mse = np.mean((X_scaled - np.mean(X_scaled, axis=0)) ** 2)

# Improvement over baseline
improvement_ratio = 1 - (mse / baseline_mse)

# Training metrics
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
best_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = np.min(history.history['val_loss'])

# Convergence metrics
convergence_speed = best_epoch / len(history.history['loss'])
avg_epoch_time = np.mean(training_metrics.epoch_times) if training_metrics.epoch_times else 0

# Per-feature MSE Histogram
per_feature_mse = np.mean((X_scaled - X_pred) ** 2, axis=0)
histogram_path = os.path.join(histograms_dir, f"per_feature_mse_hist_enc{encoding_dim}_layers{len(layer_config)}.pdf")
plt.figure(figsize=(10, 6))
plt.hist(per_feature_mse, bins=200, color='skyblue', edgecolor='black')
plt.xlabel("Per-feature MSE")
plt.ylabel("Feature Count")
plt.title(f"Distribution of Reconstruction Error (Enc {encoding_dim}, Layers {len(layer_config)})")
plt.grid(True)
plt.tight_layout()
plt.savefig(histogram_path)
plt.close()

# Per-feature Range-MSE Histogram
range_mse_histogram_path = os.path.join(histograms_dir, f"per_feature_range_mse_hist_enc{encoding_dim}_layers{len(layer_config)}.pdf")
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(per_feature_range_mse, bins=200, color='lightgreen', edgecolor='black')
plt.axvline(x=np.mean(per_feature_range_mse), color='r', linestyle='--',
           label=f'Mean Range-MSE: {np.mean(per_feature_range_mse):.6f}')
plt.axvline(x=np.median(per_feature_range_mse), color='g', linestyle='--',
           label=f'Median Range-MSE: {np.median(per_feature_range_mse):.6f}')
plt.xlabel("Per-feature Range-MSE")
plt.ylabel("Feature Count")
plt.title(f"Distribution of Range-Normalized Error (Enc {encoding_dim}, Layers {len(layer_config)})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(range_mse_histogram_path)
plt.close()

# === Step 7: Save Plots ===
# 1. Training Loss Plot
plot_path = os.path.join(plots_dir, f"loss_plot_enc{encoding_dim}_layers{len(layer_config)}.pdf")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title(f'Autoencoder Loss (Enc {encoding_dim}, Layers {len(layer_config)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

# 2. Enhanced Per-Feature MSE Histogram (already created in Step 6)

# 3. Epoch Time Plot
if training_metrics.epoch_times:
    time_plot_path = os.path.join(plots_dir, f"epoch_time_enc{encoding_dim}_layers{len(layer_config)}.pdf")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_metrics.epoch_times) + 1), training_metrics.epoch_times)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title(f'Training Time per Epoch (Enc {encoding_dim}, Layers {len(layer_config)})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(time_plot_path)
    plt.close()

# === Step 8: Log Results ===
results_path = os.path.join(results_dir, "autoencoder_results.csv")
row = {
    # Model configuration
    "encoding_dim": encoding_dim,
    "num_layers": len(layer_config),
    "layer_config": str(layer_config),
    "total_params": autoencoder.count_params(),

    # Performance metrics
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2_score": r2,
    "range_mse": range_mse,
    "mean_per_feature_range_mse": np.mean(per_feature_range_mse),
    "median_per_feature_range_mse": np.median(per_feature_range_mse),
    "baseline_mse": baseline_mse,
    "improvement_ratio": improvement_ratio,

    # Training metrics
    "final_train_loss": final_train_loss,
    "final_val_loss": final_val_loss,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "total_epochs": len(history.history['loss']),

    # Timing metrics
    "total_training_time": total_training_time,
    "avg_epoch_time": avg_epoch_time,
    "convergence_speed": convergence_speed
}

# Append to CSV
if not os.path.exists(results_path):
    pd.DataFrame([row]).to_csv(results_path, index=False)
else:
    pd.DataFrame([row]).to_csv(results_path, mode='a', header=False, index=False)

# === Step 9: Create Summary File ===
# Save a detailed summary for this specific model
summary_path = os.path.join(summaries_dir, f"model_summary_enc{encoding_dim}_layers{len(layer_config)}.txt")
with open(summary_path, 'w') as f:
    f.write(f"===== MODEL CONFIGURATION =====\n")
    f.write(f"Encoding Dimension: {encoding_dim}\n")
    f.write(f"Layer Configuration: {layer_config}\n")
    f.write(f"Total Parameters: {autoencoder.count_params():,}\n\n")

    f.write(f"===== PERFORMANCE METRICS =====\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"R² Score: {r2:.6f}\n")
    f.write(f"Range-aware MSE: {range_mse:.6f}\n")
    f.write(f"Mean Per-Feature Range-MSE: {np.mean(per_feature_range_mse):.6f}\n")
    f.write(f"Median Per-Feature Range-MSE: {np.median(per_feature_range_mse):.6f}\n")
    f.write(f"Baseline MSE: {baseline_mse:.6f}\n")
    f.write(f"Improvement over baseline: {improvement_ratio:.2%}\n\n")

    f.write(f"===== TRAINING METRICS =====\n")
    f.write(f"Final Training Loss: {final_train_loss:.6f}\n")
    f.write(f"Final Validation Loss: {final_val_loss:.6f}\n")
    f.write(f"Best Epoch: {best_epoch} (out of {len(history.history['loss'])})\n")
    f.write(f"Best Validation Loss: {best_val_loss:.6f}\n\n")

    f.write(f"===== TIMING METRICS =====\n")
    f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
    f.write(f"Average Epoch Time: {avg_epoch_time:.2f} seconds\n")
    f.write(f"Convergence Speed: {convergence_speed:.2%} (best epoch / total epochs)\n")

print(f"\n===== COMPLETED MODEL: Encoding Dim={encoding_dim}, Layers={len(layer_config)} =====")
print(f"MSE: {mse:.6f}, R²: {r2:.6f}, Improvement: {improvement_ratio:.2%}")
print(f"Range-MSE: {range_mse:.6f}")
print(f"Training Time: {total_training_time:.2f}s, Best Epoch: {best_epoch}/{len(history.history['loss'])}")
print(f"Results saved to: {results_path}")
print(f"Histograms saved to: {histograms_dir}")
print(f"Plots saved to: {plots_dir}")
print(f"Summaries saved to: {summaries_dir}")
