#!/usr/bin/env python
# Script to extract encoded features from the autoencoder model

import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model, Model
import polars as pl
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Extract encoded features from autoencoder')
    parser.add_argument('--input', type=str, default="/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/labeled_df_16_feb.csv",
                        help='Path to input CSV file')
    parser.add_argument('--models-dir', type=str, default="/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/saved_models",
                        help='Directory containing trained autoencoder models (.h5 files)')
    parser.add_argument('--scalers-dir', type=str, default="/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset",
                        help='Directory containing saved scalers (.pkl files)')
    parser.add_argument('--include-metadata', action='store_true',
                        help='Include metadata columns in the output')
    return parser.parse_args()

def process_model(model_path, input_data, scalers_dir, metadata_df=None, include_metadata=False):
    """Process a single model file and return encoded features."""
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]
    output_path = f"{model_name}.csv"

    print(f"\n{'='*80}")
    print(f"Processing model: {model_filename}")
    print(f"{'='*80}")

    # Map model names to their corresponding scaler files
    # This ensures we use the correct scaler for each model
    model_to_scaler_map = {
        "robustscaler_tuned": "robustscaler_tuned.pkl",
        "maxabsscaler_tuned": "maxabsscaler_tuned.pkl",
        "minmaxscaler_tuned": "minmaxscaler_tuned.pkl",
        "autoencoder_model_2048_standard": "robustscaler_tuned.pkl"  # Default to robustscaler for unknown models
    }

    # Try to find a matching scaler file
    scaler_path = None

    # First check if we have a direct mapping for this model
    if model_name in model_to_scaler_map:
        direct_scaler_path = os.path.join(scalers_dir, model_to_scaler_map[model_name])
        if os.path.exists(direct_scaler_path):
            scaler_path = direct_scaler_path
            print(f"Using mapped scaler for {model_name}: {os.path.basename(scaler_path)}")

    # If no direct mapping or the mapped file doesn't exist, try alternative paths
    if scaler_path is None:
        potential_scaler_paths = [
            os.path.join(scalers_dir, f"{model_name}.pkl"),
            os.path.join(scalers_dir, f"{model_name}_tuned.pkl"),
            os.path.join(scalers_dir, "robustscaler_tuned.pkl")  # Default fallback
        ]

        for path in potential_scaler_paths:
            if os.path.exists(path):
                scaler_path = path
                print(f"Using fallback scaler: {os.path.basename(scaler_path)}")
                break

    if scaler_path is None:
        print(f"⚠️ Could not find a matching scaler for {model_name}. Using data without scaling.")
        X_scaled = input_data
    else:
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Check for feature count mismatch
        try:
            # Get expected feature count from scaler
            if hasattr(scaler, 'n_features_in_'):
                expected_features = scaler.n_features_in_
            elif hasattr(scaler, 'scale_') and hasattr(scaler.scale_, 'shape'):
                expected_features = scaler.scale_.shape[0]
            else:
                expected_features = None

            # Special handling for known scalers
            scaler_basename = os.path.basename(scaler_path)
            if scaler_basename == "maxabsscaler_tuned.pkl" or scaler_basename == "minmaxscaler_tuned.pkl":
                # These scalers were trained on 97556 features
                print(f"Known scaler {scaler_basename} requires 97556 features")
                if input_data.shape[1] > 97556:
                    print(f"Trimming input data from {input_data.shape[1]} to 97556 features to match scaler.")
                    X_scaled = scaler.transform(input_data[:, :97556])
                else:
                    print(f"WARNING: Input has {input_data.shape[1]} features but scaler expects 97556. This may cause errors.")
                    X_scaled = scaler.transform(input_data)
            elif scaler_basename == "robustscaler_tuned.pkl":
                # This scaler was trained on 97614 features
                print(f"Known scaler {scaler_basename} requires 97614 features")
                if input_data.shape[1] > 97614:
                    print(f"Trimming input data from {input_data.shape[1]} to 97614 features to match scaler.")
                    X_scaled = scaler.transform(input_data[:, :97614])
                elif input_data.shape[1] < 97614:
                    print(f"WARNING: Input has {input_data.shape[1]} features but scaler expects 97614. This may cause errors.")
                    X_scaled = scaler.transform(input_data)
                else:
                    # Perfect match
                    X_scaled = scaler.transform(input_data)
            elif expected_features is not None and expected_features != input_data.shape[1]:
                # Generic handling for other scalers
                print(f"⚠️ Feature count mismatch: Scaler expects {expected_features} features, but input has {input_data.shape[1]} features.")

                if expected_features < input_data.shape[1]:
                    # Input has more features than scaler expects - trim extra features
                    print(f"Trimming input data from {input_data.shape[1]} to {expected_features} features to match scaler.")
                    X_scaled = scaler.transform(input_data[:, :expected_features])
                else:
                    # Scaler expects more features than input has - this is harder to handle
                    print(f"WARNING: Scaler expects more features than input has. This may cause errors.")
                    X_scaled = scaler.transform(input_data)
            else:
                # No mismatch or couldn't determine expected feature count
                X_scaled = scaler.transform(input_data)

        except Exception as e:
            print(f"⚠️ Error applying scaler: {str(e)}")
            print("Falling back to unscaled data.")
            X_scaled = input_data

    # Load the model
    print(f"Loading model from: {model_path}")
    full_model = load_model(model_path, compile=False)

    # Create a new model that outputs the encoded features
    # Find the bottleneck layer (typically the middle layer with the smallest dimension)
    bottleneck_layer = None
    min_units = float('inf')

    for i, layer in enumerate(full_model.layers):
        # Check if it's a Dense layer
        if 'dense' in layer.name.lower():
            # Get output shape safely - handle both older and newer TensorFlow versions
            try:
                # Try direct access first (older TF versions)
                units = layer.output_shape[-1]
            except AttributeError:
                # For newer TF versions, get shape from layer's output
                units = layer.output.shape[-1]

            if units < min_units:
                min_units = units
                bottleneck_layer = i

    if bottleneck_layer is None:
        print("Could not find bottleneck layer. Using the middle layer.")
        bottleneck_layer = len(full_model.layers) // 2
        # Need to determine min_units for the middle layer
        try:
            min_units = full_model.layers[bottleneck_layer].output_shape[-1]
        except AttributeError:
            min_units = full_model.layers[bottleneck_layer].output.shape[-1]

    print(f"Using layer {bottleneck_layer} as bottleneck with {min_units} units")

    try:
        # Create encoder model
        encoder = Model(inputs=full_model.input,
                       outputs=full_model.layers[bottleneck_layer].output)

        # Extract encoded features
        print("Extracting encoded features...")
        batch_size = 1000
        num_samples = X_scaled.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        encoded_features = np.zeros((num_samples, min_units))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            try:
                # Handle different TensorFlow versions for predict method
                if hasattr(encoder, 'predict'):
                    # Try with batch_size parameter (older TF versions)
                    try:
                        encoded_features[start_idx:end_idx] = encoder.predict(
                            X_scaled[start_idx:end_idx],
                            batch_size=min(batch_size, end_idx - start_idx),
                            verbose=0
                        )
                    except TypeError:
                        # Try without batch_size parameter (newer TF versions)
                        encoded_features[start_idx:end_idx] = encoder.predict(
                            X_scaled[start_idx:end_idx],
                            verbose=0
                        )
                else:
                    # Direct call as function (very new TF versions)
                    encoded_features[start_idx:end_idx] = encoder(X_scaled[start_idx:end_idx], training=False).numpy()

                print(f"Processed batch {i+1}/{num_batches}")
            except Exception as e:
                print(f"Error processing batch {i+1}/{num_batches}: {str(e)}")
                # Continue with next batch
    except Exception as e:
        raise Exception(f"Failed to create or use encoder model: {str(e)}")

    # Create column names for encoded features
    encoded_cols = [f"encoded_{i}" for i in range(encoded_features.shape[1])]

    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_cols)

    # Add metadata if requested
    if include_metadata and metadata_df is not None:
        # Convert polars DataFrame to pandas
        metadata_pd = metadata_df.to_pandas()
        # Combine metadata with encoded features
        final_df = pd.concat([metadata_pd, encoded_df], axis=1)
    else:
        final_df = encoded_df

    # Save to CSV using numpy to avoid pyarrow dependency
    print(f"Saving encoded features to: {output_path}")
    try:
        # First try the standard pandas method
        final_df.to_csv(output_path, index=False)
        print(f"✅ Successfully saved encoded features with shape: {final_df.shape}")
    except Exception as e:
        print(f"Warning: Error using pandas to_csv: {str(e)}")
        print(f"Trying alternative CSV export method...")

        # Manual CSV export using numpy
        try:
            # Get column names
            header = ",".join(final_df.columns)

            # Convert DataFrame to numpy array
            data_array = final_df.values

            # Save using numpy
            with open(output_path, 'w') as f:
                # Write header
                f.write(header + "\n")

                # Write data rows
                for row in data_array:
                    # Convert row values to strings and join with commas
                    row_str = ",".join(str(val) for val in row)
                    f.write(row_str + "\n")

            print(f"✅ Successfully saved encoded features with shape: {final_df.shape} using numpy method")
        except Exception as e:
            print(f"❌ Error saving CSV using alternative method: {str(e)}")
            raise

    return output_path

def main():
    args = parse_args()

    print(f"Loading data from: {args.input}")

    # Define metadata columns that are not numeric features
    non_numeric_cols = [
        "Assembly", "Domain", "Phylum", "Class", "Order", "Family",
        "Genus", "Species", "Genome accessions", "Label"
    ]

    # Load only column names first to identify numeric columns
    df_schema = pl.read_csv(args.input, n_rows=5, infer_schema_length=None)

    # Print basic dataset info
    print(f"Total columns in dataset: {len(df_schema.columns)}")

    # Check for Orthogroup138309 which needs to be excluded
    if "Orthogroup138309" in df_schema.columns:
        print(f"Note: Orthogroup138309 will be excluded as required by the models")

    # Determine which columns to load
    if args.include_metadata:
        # Load all columns except Orthogroup138309 (to match the model training)
        columns_to_load = [col for col in df_schema.columns if col != "Orthogroup138309"]
        print(f"Including metadata columns, total columns to load: {len(columns_to_load)}")
    else:
        # Load only numeric columns
        numeric_cols = [col for col, dtype in zip(df_schema.columns, df_schema.dtypes)
                        if col not in non_numeric_cols and dtype == pl.Float64]

        # Exclude Orthogroup138309 to match the model training
        numeric_cols = [col for col in numeric_cols if col != "Orthogroup138309"]
        print(f"Loading {len(numeric_cols)} numeric columns (excluding Orthogroup138309)")

        columns_to_load = numeric_cols

    # Load the data
    if args.include_metadata:
        # Load all columns
        df_full = pl.read_csv(args.input, columns=columns_to_load, infer_schema_length=None)
        print(f"Loaded full dataframe with shape: {df_full.shape}")

        # Extract metadata and numeric data
        metadata_df = df_full.select([col for col in non_numeric_cols if col in df_full.columns])
        numeric_df = df_full.select([col for col in df_full.columns if col not in non_numeric_cols])

        # Convert numeric data to numpy for processing
        X = numeric_df.fill_null(0.0).to_numpy().astype(np.float32)
    else:
        # Load only numeric columns
        df = pl.read_csv(args.input, columns=columns_to_load, infer_schema_length=None)
        X = df.fill_null(0.0).to_numpy().astype(np.float32)
        metadata_df = None

    print(f"Loaded data with shape: {X.shape}")

    # Optionally save column information for reference
    # Uncomment this block if you need to debug column issues
    """
    if args.include_metadata:
        with open("column_list.txt", "w") as f:
            f.write(f"Total columns in dataset: {len(df_schema.columns)}\n")
            f.write(f"Metadata columns ({len(metadata_df.columns)}):\n")
            for col in metadata_df.columns:
                f.write(f"{col}\n")
            f.write(f"\nNumeric columns ({len(numeric_df.columns)}):\n")
            for col in numeric_df.columns:
                f.write(f"{col}\n")
    else:
        with open("column_list.txt", "w") as f:
            f.write(f"Total columns in dataset: {len(df_schema.columns)}\n")
            f.write(f"Numeric columns ({len(df.columns)}):\n")
            for col in df.columns:
                f.write(f"{col}\n")
    """

    # Find all model files in the specified directory
    model_files = glob.glob(os.path.join(args.models_dir, "*.h5"))

    if not model_files:
        print(f"⚠️ No model files (.h5) found in {args.models_dir}")
        return

    # Filter out autoencoder_model_2048_standard.h5 as requested
    excluded_models = ["autoencoder_model_2048_standard.h5"]
    filtered_model_files = [f for f in model_files if os.path.basename(f) not in excluded_models]

    if len(filtered_model_files) < len(model_files):
        excluded_count = len(model_files) - len(filtered_model_files)
        print(f"Excluded {excluded_count} model(s): {', '.join(excluded_models)}")

    model_files = filtered_model_files
    print(f"Found {len(model_files)} model files to process")

    # Process each model file
    for model_path in model_files:
        try:
            output_path = process_model(
                model_path=model_path,
                input_data=X,
                scalers_dir=args.scalers_dir,
                metadata_df=metadata_df,
                include_metadata=args.include_metadata
            )
            print(f"✅ Successfully processed {os.path.basename(model_path)} -> {output_path}")
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(model_path)}: {str(e)}")

    print("\nAll models processed!")

if __name__ == "__main__":
    main()
