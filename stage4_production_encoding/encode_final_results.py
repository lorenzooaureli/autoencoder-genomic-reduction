#!/usr/bin/env python3
"""
Script to encode data using a saved autoencoder model.

This script:
1. Loads data from an input CSV file
2. Applies a RobustScaler from a .pkl file
3. Uses the encoder part of an autoencoder model (.h5 file) to generate encoded features
4. Saves the encoded features to a CSV file

Usage:
    python encode_data.py <input_file> <scaler_path> <model_path> [output_file]

Arguments:
    input_file: Path to the input CSV file
    scaler_path: Path to the saved scaler (.pkl file)
    model_path: Path to the saved autoencoder model (.h5 file)
    output_file: (Optional) Path to save the encoded features (default: encoded_output.csv)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
from tensorflow.keras.models import load_model, Model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Encode data using a saved autoencoder model')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('scaler_path', help='Path to the saved scaler (.pkl file)')
    parser.add_argument('model_path', help='Path to the saved autoencoder model (.h5 file)')
    parser.add_argument('output_file', nargs='?', default='encoded_output.csv',
                        help='Path to save the encoded features (default: encoded_output.csv)')
    return parser.parse_args()

# Parse command-line arguments
args = parse_arguments()
input_file = args.input_file
scaler_path = args.scaler_path
model_path = args.model_path
output_file = args.output_file

# Define metadata columns based on the original dataset
metadata_cols = [
    "Assembly", "Domain", "Phylum", "Class", "Order", "Family",
    "Genus", "Species", "Genome accessions", "Label"
]

print(f"Encoding data from {input_file} using the saved autoencoder model...")

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} does not exist.")
    sys.exit(1)

# Check if the scaler file exists
if not os.path.exists(scaler_path):
    print(f"Error: Scaler file {scaler_path} does not exist.")
    sys.exit(1)

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} does not exist.")
    sys.exit(1)

# Load the data
try:
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Successfully loaded data with shape: {df.shape}")

    # Extract metadata columns that exist in the dataframe
    available_metadata_cols = [col for col in metadata_cols if col in df.columns]
    metadata_df = df[available_metadata_cols]
    print(f"Extracted {len(available_metadata_cols)} metadata columns.")

    # Extract feature columns (excluding metadata)
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Exclude Orthogroup138309 if present
    if "Orthogroup138309" in feature_cols:
        feature_cols.remove("Orthogroup138309")
        print("Excluded Orthogroup138309 column.")

    # Extract features
    X = df[feature_cols].values.astype(np.float32)
    print(f"Extracted features with shape: {X.shape}")

except Exception as e:
    print(f"Error loading data: {str(e)}")
    sys.exit(1)

# Load the scaler
try:
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Successfully loaded scaler.")

    # Scale the data
    X_scaled = scaler.transform(X)
    print(f"Scaled data shape: {X_scaled.shape}")

except Exception as e:
    print(f"Error loading or applying scaler: {str(e)}")
    sys.exit(1)

# Load the model and create encoder
try:
    print(f"Loading model from {model_path}...")
    full_model = load_model(model_path, compile=False)
    print("Successfully loaded model.")

    # Find the bottleneck layer (the layer with the smallest dimension)
    bottleneck_layer = None
    min_units = float('inf')

    for i, layer in enumerate(full_model.layers):
        if 'dense' in layer.name.lower():
            try:
                units = layer.output_shape[-1]
            except AttributeError:
                units = layer.output.shape[-1]

            if units < min_units:
                min_units = units
                bottleneck_layer = i

    if bottleneck_layer is None:
        print("Error: Could not find bottleneck layer in the model.")
        sys.exit(1)

    print(f"Found bottleneck layer at index {bottleneck_layer} with {min_units} units.")

    # Create encoder model
    encoder = Model(inputs=full_model.input,
                    outputs=full_model.layers[bottleneck_layer].output)
    print("Created encoder model.")

    # Generate encoded features
    print("Generating encoded features...")
    encoded_features = encoder.predict(X_scaled)
    print(f"Encoded features shape: {encoded_features.shape}")

except Exception as e:
    print(f"Error loading model or generating encoded features: {str(e)}")
    sys.exit(1)

# Create output dataframe and save
try:
    # Create column names for encoded features
    encoded_cols = [f"encoded_{i}" for i in range(encoded_features.shape[1])]

    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_cols)

    # Add metadata
    final_df = pd.concat([metadata_df.reset_index(drop=True), encoded_df], axis=1)

    # Save to CSV
    print(f"Saving encoded features to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print(f"Successfully saved encoded features to {output_file}.")

    # Print a sample of the encoded features
    print("\nSample of encoded features (first 5 columns):")
    print(encoded_df.iloc[:, :5])

except Exception as e:
    print(f"Error saving encoded features: {str(e)}")
    sys.exit(1)

print("\nEncoding process completed successfully!")
