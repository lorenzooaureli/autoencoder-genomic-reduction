#!/usr/bin/env python3

'''
Scaler Feature Analysis

This script analyzes how different scaling methods affect feature distributions in a dataset.
It applies various scikit-learn scalers (StandardScaler, RobustScaler, PowerTransformer, etc.)
to the input data and generates histograms and statistics for feature ranges, means, and
standard deviations after scaling. Results are saved as PDFs and CSVs for comparison.
'''

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    MaxAbsScaler, MinMaxScaler, QuantileTransformer, minmax_scale
)
import os

# === CONFIG ===
INPUT_CSV = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/labeled_df_16_feb.csv"
OUTPUT_DIR = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset"
META_COLS = ["Assembly", "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Genome accessions", "Label"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pl.read_csv(INPUT_CSV)
X = df.drop(META_COLS).to_numpy()

# === DEFINE SCALERS ===
scalers = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "PowerTransformer": PowerTransformer(),
    "MaxAbsScaler": MaxAbsScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "QuantileTransformer": QuantileTransformer(output_distribution="normal"),
    "minmax_scale": "function"
}

# === HELPER FUNCTION ===
def create_histogram_and_csv(data, title, filename_prefix):
    import matplotlib.pyplot as plt
    counts, bins, patches = plt.hist(data, bins=300, edgecolor='black')
    bin_width = bins[1] - bins[0]
    plt.xlim(bins[0] - bin_width / 2, bins[-1] + bin_width / 2)
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {title} Across Features")
    plt.tight_layout()
    offset = max(counts) * 0.01
    for count, bin_left, patch in zip(counts, bins, patches):
        if count > 0:
            plt.text(
                bin_left + patch.get_width() / 2,
                count + offset,
                f"{int(count):,}",
                ha='center',
                va='bottom',
                fontsize=7,
                rotation=90
            )
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.close()

    bin_data = pd.DataFrame({
        "bin_start": bins[:-1],
        "bin_end": bins[1:],
        "count": counts.astype(int)
    })
    csv_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_bins.csv")
    bin_data.to_csv(csv_path, index=False)
    return pdf_path, csv_path

# === MAIN LOOP ===
results = {}
for name, scaler in scalers.items():
    try:
        if name == "minmax_scale":
            X_scaled = minmax_scale(X, axis=0)
        else:
            X_scaled = scaler.fit_transform(X)

        feature_range = np.max(X_scaled, axis=0) - np.min(X_scaled, axis=0)
        feature_mean = np.mean(X_scaled, axis=0)
        feature_std = np.std(X_scaled, axis=0)

        results[name] = {
            "range": create_histogram_and_csv(feature_range, "Feature Range", f"{name}_range_histogram"),
            "mean": create_histogram_and_csv(feature_mean, "Feature Mean", f"{name}_mean_histogram"),
            "std": create_histogram_and_csv(feature_std, "Feature Std Dev", f"{name}_std_histogram"),
        }

        print(f"✅ Finished: {name}")
    except Exception as e:
        print(f"❌ Skipping {name} due to error: {e}")
        results[name] = {"error": str(e)}

print("✅ All scalers processed.")

