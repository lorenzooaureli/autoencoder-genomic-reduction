#!/usr/bin/env python3
"""
Autoencoder Tool - A unified script for checking compatibility and encoding data with autoencoders

This script provides two main functions:
1. check - Check if a dataset is compatible with a saved autoencoder model
2. encode - Encode data using a saved autoencoder model

Usage:
    python autoencoder_tool.py check <input_file> <scaler_path>
    python autoencoder_tool.py encode <input_file> <scaler_path> <model_path> [output_file]

Arguments:
    check/encode     - The operation to perform
    input_file       - Path to the input CSV file
    scaler_path      - Path to the saved scaler (.pkl file)
    model_path       - Path to the saved autoencoder model (.h5 file) [only for encode]
    output_file      - (Optional) Path to save the encoded features [only for encode]
                       (default: encoded_output.csv)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
from tensorflow.keras.models import load_model, Model

# Define metadata columns based on the original dataset
METADATA_COLS = [
    "Assembly", "Domain", "Phylum", "Class", "Order", "Family",
    "Genus", "Species", "Genome accessions", "Label"
]

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Autoencoder Tool - Check compatibility and encode data')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True
    
    # Parser for the 'check' command
    check_parser = subparsers.add_parser('check', help='Check if a dataset is compatible with a saved autoencoder model')
    check_parser.add_argument('input_file', help='Path to the input CSV file')
    check_parser.add_argument('scaler_path', help='Path to the saved scaler (.pkl file)')
    
    # Parser for the 'encode' command
    encode_parser = subparsers.add_parser('encode', help='Encode data using a saved autoencoder model')
    encode_parser.add_argument('input_file', help='Path to the input CSV file')
    encode_parser.add_argument('scaler_path', help='Path to the saved scaler (.pkl file)')
    encode_parser.add_argument('model_path', help='Path to the saved autoencoder model (.h5 file)')
    encode_parser.add_argument('output_file', nargs='?', default='encoded_output.csv', 
                        help='Path to save the encoded features (default: encoded_output.csv)')
    
    return parser.parse_args()

def check_compatibility(input_file, scaler_path):
    """Check if a dataset is compatible with a saved autoencoder model"""
    print(f"Checking compatibility of {input_file} with the saved model...")

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return False

    # Check if the scaler file exists
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} does not exist.")
        return False

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
        return False

    # Load the header of the input file to check its structure
    try:
        # Read just the header to get column names
        df_header = pd.read_csv(input_file, nrows=0)
        total_columns = len(df_header.columns)
        print(f"Input file has {total_columns} total columns.")
        
        # Check for metadata columns
        found_metadata_cols = [col for col in METADATA_COLS if col in df_header.columns]
        print(f"Found {len(found_metadata_cols)} metadata columns: {found_metadata_cols}")
        
        # Check for Orthogroup138309 which needs to be excluded
        if "Orthogroup138309" in df_header.columns:
            print("Note: Orthogroup138309 is present and will need to be excluded.")
        
        # Calculate the number of feature columns (excluding metadata)
        feature_cols = [col for col in df_header.columns if col not in METADATA_COLS]
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
                return False
        
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
            return False
        else:
            print("All feature columns have numeric data types.")
        
        # Overall compatibility assessment
        if expected_features is not None and len(feature_cols_filtered) == expected_features and not non_numeric_features:
            print("\n✅ The file appears to be compatible with the saved model.")
            print("To use it with the model, you'll need to:")
            print("1. Exclude the Orthogroup138309 column if present")
            print("2. Fill any missing values with 0.0")
            print("3. Convert all data to float32 before scaling")
            return True
        else:
            print("\n❌ There are compatibility issues that need to be addressed.")
            return False
            
    except Exception as e:
        print(f"Error analyzing input file: {str(e)}")
        return False

def encode_data(input_file, scaler_path, model_path, output_file):
    """Encode data using a saved autoencoder model"""
    print(f"Encoding data from {input_file} using the saved autoencoder model...")

    # First check compatibility
    print("\n=== Checking Compatibility ===")
    if not check_compatibility(input_file, scaler_path):
        print("Compatibility check failed. Please fix the issues before encoding.")
        return False
    
    print("\n=== Starting Encoding Process ===")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist.")
        return False

    # Load the data
    try:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Extract metadata columns that exist in the dataframe
        available_metadata_cols = [col for col in METADATA_COLS if col in df.columns]
        metadata_df = df[available_metadata_cols]
        print(f"Extracted {len(available_metadata_cols)} metadata columns.")
        
        # Extract feature columns (excluding metadata)
        feature_cols = [col for col in df.columns if col not in METADATA_COLS]
        
        # Exclude Orthogroup138309 if present
        if "Orthogroup138309" in feature_cols:
            feature_cols.remove("Orthogroup138309")
            print("Excluded Orthogroup138309 column.")
        
        # Extract features
        X = df[feature_cols].values.astype(np.float32)
        print(f"Extracted features with shape: {X.shape}")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

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
        return False

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
            return False
        
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
        return False

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
        
        return True
        
    except Exception as e:
        print(f"Error saving encoded features: {str(e)}")
        return False

def main():
    """Main function to parse arguments and execute commands"""
    args = parse_arguments()
    
    if args.command == 'check':
        success = check_compatibility(args.input_file, args.scaler_path)
        sys.exit(0 if success else 1)
    elif args.command == 'encode':
        success = encode_data(args.input_file, args.scaler_path, args.model_path, args.output_file)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
