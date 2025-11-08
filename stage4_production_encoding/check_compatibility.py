#!/usr/bin/env python3
"""
Script to check if a dataset is compatible with a saved autoencoder model.

Usage:
    python check_compatibility.py <input_file> <scaler_path>

Arguments:
    input_file: Path to the input CSV file to check
    scaler_path: Path to the saved scaler (.pkl file)
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check compatibility of a dataset with a saved autoencoder model')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('scaler_path', help='Path to the saved scaler (.pkl file)')
    return parser.parse_args()

# Parse command-line arguments
args = parse_arguments()
input_file = args.input_file
scaler_path = args.scaler_path

# Define metadata columns based on the original dataset
metadata_cols = [
    "Assembly", "Domain", "Phylum", "Class", "Order", "Family",
    "Genus", "Species", "Genome accessions", "Label"
]

print(f"Checking compatibility of {input_file} with the saved model...")

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} does not exist.")
    sys.exit(1)

# Check if the scaler file exists
if not os.path.exists(scaler_path):
    print(f"Error: Scaler file {scaler_path} does not exist.")
    sys.exit(1)

# Load the scaler to check its properties
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    if hasattr(scaler, 'n_features_in_'):
        expected_features = scaler.n_features_in_
        print(f"The model expects {expected_features} input features.")
    else:
        print("Warning: Could not determine the number of expected features from the scaler.")
        expected_features = None
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    sys.exit(1)

# Load the header of the input file to check its structure
try:
    # Read just the header to get column names
    df_header = pd.read_csv(input_file, nrows=0)
    total_columns = len(df_header.columns)
    print(f"Input file has {total_columns} total columns.")

    # Check for metadata columns
    found_metadata_cols = [col for col in metadata_cols if col in df_header.columns]
    print(f"Found {len(found_metadata_cols)} metadata columns: {found_metadata_cols}")

    # Check for Orthogroup138309 which needs to be excluded
    if "Orthogroup138309" in df_header.columns:
        print("Note: Orthogroup138309 is present and will need to be excluded.")

    # Calculate the number of feature columns (excluding metadata)
    feature_cols = [col for col in df_header.columns if col not in metadata_cols]
    print(f"Found {len(feature_cols)} potential feature columns.")

    # Calculate the number of feature columns after excluding Orthogroup138309
    feature_cols_filtered = [col for col in feature_cols if col != "Orthogroup138309"]
    print(f"After excluding Orthogroup138309: {len(feature_cols_filtered)} feature columns.")

    # Check if the number of features matches what the model expects
    if expected_features is not None:
        if len(feature_cols_filtered) == expected_features:
            print(f"✅ The number of features ({len(feature_cols_filtered)}) matches what the model expects ({expected_features}).")
        else:
            print(f"❌ The number of features ({len(feature_cols_filtered)}) does NOT match what the model expects ({expected_features}).")

            # Provide more details about the mismatch
            if len(feature_cols_filtered) < expected_features:
                print(f"   Missing {expected_features - len(feature_cols_filtered)} features.")
            else:
                print(f"   Has {len(feature_cols_filtered) - expected_features} extra features.")

    # Now try to read the actual data to check for any issues
    print("\nReading the data to check for potential issues...")
    df = pd.read_csv(input_file)
    print(f"Successfully read the data. Found {len(df)} rows.")

    # Check for missing values in feature columns
    feature_cols_in_df = [col for col in feature_cols_filtered if col in df.columns]
    missing_values = df[feature_cols_in_df].isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values in the feature columns.")
        print("These will be filled with 0.0 during processing.")
    else:
        print("No missing values found in the feature columns.")

    # Check data types
    non_numeric_features = [col for col in feature_cols_in_df if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_features:
        print(f"Warning: Found {len(non_numeric_features)} non-numeric feature columns.")
        print(f"First few non-numeric columns: {non_numeric_features[:5]}")
    else:
        print("All feature columns have numeric data types.")

    # Overall compatibility assessment
    if expected_features is not None and len(feature_cols_filtered) == expected_features and not non_numeric_features:
        print("\n✅ The file appears to be compatible with the saved model.")
        print("To use it with the model, you'll need to:")
        print("1. Exclude the Orthogroup138309 column if present")
        print("2. Fill any missing values with 0.0")
        print("3. Convert all data to float32 before scaling")
    else:
        print("\n❌ There are compatibility issues that need to be addressed.")

except Exception as e:
    print(f"Error analyzing input file: {str(e)}")
    sys.exit(1)
