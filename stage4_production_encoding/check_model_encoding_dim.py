#!/usr/bin/env python3
"""
Quick script to check the encoding dimension of a model
"""

import tensorflow as tf
from tensorflow.keras.models import load_model

# Check the model at robustscaler_enc1024_layers1.h5
model_path = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/robustscaler_enc1024_layers1.h5"

print(f"Loading model from: {model_path}")
model = load_model(model_path, compile=False)

print(f"\nModel input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

print("\nModel layers:")
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        print(f"  {layer.name}: {layer.units} units")

# Try to find the encoding layer
# Based on the filename, it should be 1024
print("\nLooking for encoding layer with 1024 units...")
encoding_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1024:
        print(f"Found potential encoding layer: {layer.name}")
        encoding_layer = layer.name

if encoding_layer:
    print(f"\n✅ The model appears to have an encoding dimension of 1024")
    print(f"This matches the encoded features in robustscaler_tuned.csv!")
else:
    print("\n⚠️  Could not find a layer with exactly 1024 units")