#!/usr/bin/env python
# Script to check the scaler properties

import pickle
import numpy as np

# Path to the scaler
scaler_path = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/robustscaler_tuned.pkl"

print(f"Loading scaler from {scaler_path}...")
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

# Check center_ attribute
if hasattr(scaler, 'center_'):
    print(f"Shape of center_ attribute: {scaler.center_.shape}")

# Print other attributes
print("\nOther attributes:")
for attr in dir(scaler):
    if not attr.startswith('_') and attr not in ['n_features_in_', 'scale_', 'center_', 'fit', 'transform', 'fit_transform', 'inverse_transform']:
        try:
            value = getattr(scaler, attr)
            if not callable(value):
                print(f"  {attr}: {value}")
        except:
            print(f"  {attr}: <error accessing>")

print("\nDone!")
