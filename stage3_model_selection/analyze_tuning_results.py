#!/usr/bin/env python3
"""
PowerTransformer Tuning Results Analysis

This script analyzes the results of hyperparameter tuning for autoencoders
with PowerTransformer scaling. It generates visualizations and summaries to help
identify the best hyperparameter combinations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

# === Configuration ===
RESULTS_PATH = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/autoencoder_results.csv"
OUTPUT_DIR = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/analysis"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Results ===
if not os.path.exists(RESULTS_PATH):
    print(f"Error: Results file not found at {RESULTS_PATH}")
    print("Run the tuning jobs first to generate results.")
    exit(1)

results = pd.read_csv(RESULTS_PATH)
print(f"Loaded {len(results)} model configurations")

# === Data Preparation ===
# Convert layer_config from string to actual list
if 'layer_config' in results.columns:
    results['layer_config'] = results['layer_config'].apply(literal_eval)
    # Add a column for the deepest layer size
    results['deepest_layer'] = results['layer_config'].apply(lambda x: min(x) if x else 0)

# === Best Models ===
# Find best models by different metrics
best_mse = results.loc[results['mse'].idxmin()]
best_r2 = results.loc[results['r2_score'].idxmax()]
best_improvement = results.loc[results['improvement_ratio'].idxmax()]
fastest_convergence = results.loc[results['convergence_speed'].idxmin()]

# === Generate Summary Report ===
summary_path = os.path.join(OUTPUT_DIR, "tuning_summary.txt")
with open(summary_path, 'w') as f:
    f.write("===== POWERTRANSFORMER AUTOENCODER TUNING SUMMARY =====\n\n")
    
    f.write("=== OVERALL STATISTICS ===\n")
    f.write(f"Total models evaluated: {len(results)}\n")
    f.write(f"MSE range: {results['mse'].min():.6f} to {results['mse'].max():.6f}\n")
    f.write(f"R² range: {results['r2_score'].min():.6f} to {results['r2_score'].max():.6f}\n")
    f.write(f"Improvement ratio range: {results['improvement_ratio'].min():.2%} to {results['improvement_ratio'].max():.2%}\n\n")
    
    f.write("=== BEST MODEL BY MSE ===\n")
    f.write(f"Encoding dimension: {best_mse['encoding_dim']}\n")
    f.write(f"Layer configuration: {best_mse['layer_config']}\n")
    f.write(f"MSE: {best_mse['mse']:.6f}\n")
    f.write(f"R²: {best_mse['r2_score']:.6f}\n")
    f.write(f"Improvement ratio: {best_mse['improvement_ratio']:.2%}\n\n")
    
    f.write("=== BEST MODEL BY R² ===\n")
    f.write(f"Encoding dimension: {best_r2['encoding_dim']}\n")
    f.write(f"Layer configuration: {best_r2['layer_config']}\n")
    f.write(f"MSE: {best_r2['mse']:.6f}\n")
    f.write(f"R²: {best_r2['r2_score']:.6f}\n")
    f.write(f"Improvement ratio: {best_r2['improvement_ratio']:.2%}\n\n")
    
    f.write("=== BEST MODEL BY IMPROVEMENT RATIO ===\n")
    f.write(f"Encoding dimension: {best_improvement['encoding_dim']}\n")
    f.write(f"Layer configuration: {best_improvement['layer_config']}\n")
    f.write(f"MSE: {best_improvement['mse']:.6f}\n")
    f.write(f"R²: {best_improvement['r2_score']:.6f}\n")
    f.write(f"Improvement ratio: {best_improvement['improvement_ratio']:.2%}\n\n")
    
    f.write("=== FASTEST CONVERGING MODEL ===\n")
    f.write(f"Encoding dimension: {fastest_convergence['encoding_dim']}\n")
    f.write(f"Layer configuration: {fastest_convergence['layer_config']}\n")
    f.write(f"Best epoch: {fastest_convergence['best_epoch']} (out of {fastest_convergence['total_epochs']})\n")
    f.write(f"MSE: {fastest_convergence['mse']:.6f}\n")
    f.write(f"Training time: {fastest_convergence['total_training_time']:.2f} seconds\n\n")
    
    f.write("=== ENCODING DIMENSION ANALYSIS ===\n")
    for dim in sorted(results['encoding_dim'].unique()):
        subset = results[results['encoding_dim'] == dim]
        f.write(f"Dimension {dim}:\n")
        f.write(f"  Count: {len(subset)}\n")
        f.write(f"  Avg MSE: {subset['mse'].mean():.6f}\n")
        f.write(f"  Avg R²: {subset['r2_score'].mean():.6f}\n")
        f.write(f"  Best MSE: {subset['mse'].min():.6f}\n\n")
    
    f.write("=== LAYER COUNT ANALYSIS ===\n")
    for layers in sorted(results['num_layers'].unique()):
        subset = results[results['num_layers'] == layers]
        f.write(f"{layers} layers:\n")
        f.write(f"  Count: {len(subset)}\n")
        f.write(f"  Avg MSE: {subset['mse'].mean():.6f}\n")
        f.write(f"  Avg R²: {subset['r2_score'].mean():.6f}\n")
        f.write(f"  Best MSE: {subset['mse'].min():.6f}\n\n")

print(f"Summary report saved to {summary_path}")

# === Visualizations ===

# 1. MSE by Encoding Dimension and Layer Count
plt.figure(figsize=(12, 8))
sns.boxplot(x='encoding_dim', y='mse', hue='num_layers', data=results)
plt.title('MSE by Encoding Dimension and Layer Count')
plt.xlabel('Encoding Dimension')
plt.ylabel('Mean Squared Error (lower is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mse_by_encoding_dim_and_layers.pdf'))
plt.close()

# 2. R² by Encoding Dimension and Layer Count
plt.figure(figsize=(12, 8))
sns.boxplot(x='encoding_dim', y='r2_score', hue='num_layers', data=results)
plt.title('R² Score by Encoding Dimension and Layer Count')
plt.xlabel('Encoding Dimension')
plt.ylabel('R² Score (higher is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'r2_by_encoding_dim_and_layers.pdf'))
plt.close()

# 3. Training Time by Model Complexity
plt.figure(figsize=(12, 8))
results['model_size'] = results['encoding_dim'] * results['num_layers']
sns.scatterplot(x='total_params', y='total_training_time', 
                hue='encoding_dim', size='num_layers', 
                sizes=(50, 200), alpha=0.7, data=results)
plt.title('Training Time by Model Complexity')
plt.xlabel('Total Parameters')
plt.ylabel('Training Time (seconds)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_time_by_complexity.pdf'))
plt.close()

# 4. Performance vs. Convergence Speed
plt.figure(figsize=(12, 8))
sns.scatterplot(x='convergence_speed', y='mse', 
                hue='encoding_dim', size='num_layers',
                sizes=(50, 200), alpha=0.7, data=results)
plt.title('Performance vs. Convergence Speed')
plt.xlabel('Convergence Speed (lower is faster)')
plt.ylabel('MSE (lower is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_vs_convergence.pdf'))
plt.close()

# 5. Heatmap of MSE by Encoding Dimension and Layer Count
pivot_mse = results.pivot_table(
    values='mse', 
    index='num_layers',
    columns='encoding_dim',
    aggfunc='mean'
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_mse, annot=True, fmt='.6f', cmap='viridis_r')
plt.title('Mean MSE by Encoding Dimension and Layer Count')
plt.ylabel('Number of Layers')
plt.xlabel('Encoding Dimension')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mse_heatmap.pdf'))
plt.close()

# 6. Improvement Ratio by Configuration
plt.figure(figsize=(12, 8))
sns.barplot(x='encoding_dim', y='improvement_ratio', hue='num_layers', data=results)
plt.title('Improvement Ratio by Configuration')
plt.xlabel('Encoding Dimension')
plt.ylabel('Improvement Ratio (higher is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'improvement_ratio.pdf'))
plt.close()

print(f"All analysis plots saved to {OUTPUT_DIR}")

# === Create Recommendations ===
recommendations_path = os.path.join(OUTPUT_DIR, "recommendations.txt")
with open(recommendations_path, 'w') as f:
    f.write("===== POWERTRANSFORMER AUTOENCODER RECOMMENDATIONS =====\n\n")
    
    f.write("Based on the hyperparameter tuning results, here are the recommended configurations:\n\n")
    
    f.write("1. BEST OVERALL PERFORMANCE:\n")
    if best_mse['encoding_dim'] == best_r2['encoding_dim'] and best_mse['num_layers'] == best_r2['num_layers']:
        f.write(f"   Encoding dimension: {best_mse['encoding_dim']}\n")
        f.write(f"   Layer configuration: {best_mse['layer_config']}\n")
        f.write(f"   MSE: {best_mse['mse']:.6f}, R²: {best_mse['r2_score']:.6f}\n")
        f.write(f"   This configuration is optimal for both MSE and R² metrics.\n\n")
    else:
        f.write(f"   Encoding dimension: {best_mse['encoding_dim']}\n")
        f.write(f"   Layer configuration: {best_mse['layer_config']}\n")
        f.write(f"   MSE: {best_mse['mse']:.6f}, R²: {best_mse['r2_score']:.6f}\n")
        f.write(f"   This configuration provides the lowest reconstruction error.\n\n")
    
    f.write("2. BALANCED PERFORMANCE AND TRAINING SPEED:\n")
    # Find models with good performance and reasonable training time
    balanced_models = results[
        (results['mse'] < results['mse'].median()) & 
        (results['total_training_time'] < results['total_training_time'].median())
    ]
    if len(balanced_models) > 0:
        best_balanced = balanced_models.loc[balanced_models['mse'].idxmin()]
        f.write(f"   Encoding dimension: {best_balanced['encoding_dim']}\n")
        f.write(f"   Layer configuration: {best_balanced['layer_config']}\n")
        f.write(f"   MSE: {best_balanced['mse']:.6f}, Training time: {best_balanced['total_training_time']:.2f} seconds\n")
        f.write(f"   This configuration balances good performance with reasonable training time.\n\n")
    else:
        f.write("   No configuration found that balances performance and training speed well.\n\n")
    
    f.write("3. FASTEST USABLE MODEL:\n")
    # Find models with acceptable performance and fast training
    threshold = results['mse'].median() * 1.1  # Allow up to 10% worse than median MSE
    fast_models = results[results['mse'] < threshold]
    if len(fast_models) > 0:
        fastest = fast_models.loc[fast_models['total_training_time'].idxmin()]
        f.write(f"   Encoding dimension: {fastest['encoding_dim']}\n")
        f.write(f"   Layer configuration: {fastest['layer_config']}\n")
        f.write(f"   MSE: {fastest['mse']:.6f}, Training time: {fastest['total_training_time']:.2f} seconds\n")
        f.write(f"   This configuration prioritizes speed while maintaining acceptable performance.\n\n")
    else:
        f.write("   No configuration found that offers fast training with acceptable performance.\n\n")
    
    f.write("4. GENERAL OBSERVATIONS:\n")
    # Encoding dimension trends
    enc_dim_mse = results.groupby('encoding_dim')['mse'].mean().sort_values()
    best_enc_dim = enc_dim_mse.index[0]
    worst_enc_dim = enc_dim_mse.index[-1]
    f.write(f"   - Encoding dimension: {best_enc_dim} tends to give the best average performance.\n")
    f.write(f"   - Encoding dimension: {worst_enc_dim} tends to give the worst average performance.\n")
    
    # Layer count trends
    layer_mse = results.groupby('num_layers')['mse'].mean().sort_values()
    best_layers = layer_mse.index[0]
    worst_layers = layer_mse.index[-1]
    f.write(f"   - Models with {best_layers} layers tend to perform best on average.\n")
    f.write(f"   - Models with {worst_layers} layers tend to perform worst on average.\n")
    
    # Training time observations
    if results['total_training_time'].corr(results['num_layers']) > 0.5:
        f.write("   - Training time increases significantly with the number of layers.\n")
    if results['total_training_time'].corr(results['encoding_dim']) > 0.5:
        f.write("   - Training time increases significantly with encoding dimension.\n")
    
    f.write("\n5. NEXT STEPS:\n")
    f.write("   - Consider running a more focused hyperparameter search around the best configurations.\n")
    f.write(f"   - Test the best model(s) on validation data to confirm performance.\n")
    f.write("   - Compare these results with other scaling methods to determine the best approach.\n")

print(f"Recommendations saved to {recommendations_path}")
print("Analysis complete!")
