#!/usr/bin/env python
# Script to check all scalers

import pickle
import os

# Base directory for scalers
base_dir = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset"

# List of scalers to check
scalers = [
    "robustscaler_tuned.pkl",
    "maxabsscaler_tuned.pkl",
    "minmaxscaler_tuned.pkl"
]

for scaler_file in scalers:
    scaler_path = os.path.join(base_dir, scaler_file)
    
    if not os.path.exists(scaler_path):
        print(f"Scaler not found: {scaler_path}")
        continue
    
    print(f"\nChecking scaler: {scaler_file}")
    print("-" * 50)
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Print scaler info
        print(f"Scaler type: {type(scaler).__name__}")
        
        # Check for n_features_in_ attribute
        if hasattr(scaler, 'n_features_in_'):
            print(f"Number of features scaler was trained on (n_features_in_): {scaler.n_features_in_}")
        
        # Check scale_ attribute
        if hasattr(scaler, 'scale_'):
            print(f"Shape of scale_ attribute: {scaler.scale_.shape}")
        
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")

print("\nDone!")
