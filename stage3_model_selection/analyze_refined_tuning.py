#!/usr/bin/env python3
"""
Refined PowerTransformer Tuning Results Analysis

This script analyzes the results of the refined hyperparameter tuning for autoencoders
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
RESULTS_PATH = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/refined_tuning/refined_autoencoder_results.csv"
ORIGINAL_RESULTS_PATH = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/autoencoder_results.csv"
OUTPUT_DIR = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/refined_tuning/analysis"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Results ===
if not os.path.exists(RESULTS_PATH):
    print(f"Error: Results file not found at {RESULTS_PATH}")
    print("Run the refined tuning jobs first to generate results.")
    exit(1)

results = pd.read_csv(RESULTS_PATH)
print(f"Loaded {len(results)} refined model configurations")

# Load original results for comparison
if os.path.exists(ORIGINAL_RESULTS_PATH):
    original_results = pd.read_csv(ORIGINAL_RESULTS_PATH)
    print(f"Loaded {len(original_results)} original model configurations for comparison")
    # Add a source column to distinguish between original and refined results
    original_results['source'] = 'original'
    results['source'] = 'refined'
    
    # Add default values for new parameters in original results
    if 'learning_rate' not in original_results.columns:
        original_results['learning_rate'] = 0.001
    if 'batch_size' not in original_results.columns:
        original_results['batch_size'] = 16
    if 'dropout_rate' not in original_results.columns:
        original_results['dropout_rate'] = 0.2
    
    # Combine results for comparison
    combined_results = pd.concat([original_results, results], ignore_index=True)
else:
    print("Original results file not found. Proceeding with refined results only.")
    combined_results = results.copy()
    combined_results['source'] = 'refined'

# === Data Preparation ===
# Convert layer_config from string to actual list
if 'layer_config' in results.columns:
    results['layer_config'] = results['layer_config'].apply(literal_eval)
    results['deepest_layer'] = results['layer_config'].apply(lambda x: min(x) if x else 0)

# === Best Models ===
# Find best models by different metrics
best_mse = results.loc[results['mse'].idxmin()]
best_r2 = results.loc[results['r2_score'].idxmax()]
best_improvement = results.loc[results['improvement_ratio'].idxmax()]
fastest_convergence = results.loc[results['convergence_speed'].idxmin()]

# === Generate Summary Report ===
summary_path = os.path.join(OUTPUT_DIR, "refined_tuning_summary.txt")
with open(summary_path, 'w') as f:
    f.write("===== REFINED POWERTRANSFORMER AUTOENCODER TUNING SUMMARY =====\n\n")
    
    f.write("=== OVERALL STATISTICS ===\n")
    f.write(f"Total models evaluated: {len(results)}\n")
    f.write(f"MSE range: {results['mse'].min():.6f} to {results['mse'].max():.6f}\n")
    f.write(f"R² range: {results['r2_score'].min():.6f} to {results['r2_score'].max():.6f}\n")
    f.write(f"Improvement ratio range: {results['improvement_ratio'].min():.2%} to {results['improvement_ratio'].max():.2%}\n\n")
    
    f.write("=== BEST MODEL BY MSE ===\n")
    f.write(f"Encoding dimension: {best_mse['encoding_dim']}\n")
    f.write(f"Layer configuration: {best_mse['layer_config']}\n")
    f.write(f"Learning rate: {best_mse['learning_rate']}\n")
    f.write(f"Batch size: {best_mse['batch_size']}\n")
    f.write(f"Dropout rate: {best_mse['dropout_rate']}\n")
    f.write(f"MSE: {best_mse['mse']:.6f}\n")
    f.write(f"R²: {best_mse['r2_score']:.6f}\n")
    f.write(f"Improvement ratio: {best_mse['improvement_ratio']:.2%}\n\n")
    
    f.write("=== BEST MODEL BY R² ===\n")
    f.write(f"Encoding dimension: {best_r2['encoding_dim']}\n")
    f.write(f"Layer configuration: {best_r2['layer_config']}\n")
    f.write(f"Learning rate: {best_r2['learning_rate']}\n")
    f.write(f"Batch size: {best_r2['batch_size']}\n")
    f.write(f"Dropout rate: {best_r2['dropout_rate']}\n")
    f.write(f"MSE: {best_r2['mse']:.6f}\n")
    f.write(f"R²: {best_r2['r2_score']:.6f}\n")
    f.write(f"Improvement ratio: {best_r2['improvement_ratio']:.2%}\n\n")
    
    f.write("=== FASTEST CONVERGING MODEL ===\n")
    f.write(f"Encoding dimension: {fastest_convergence['encoding_dim']}\n")
    f.write(f"Layer configuration: {fastest_convergence['layer_config']}\n")
    f.write(f"Learning rate: {fastest_convergence['learning_rate']}\n")
    f.write(f"Batch size: {fastest_convergence['batch_size']}\n")
    f.write(f"Dropout rate: {fastest_convergence['dropout_rate']}\n")
    f.write(f"Best epoch: {fastest_convergence['best_epoch']} (out of {fastest_convergence['total_epochs']})\n")
    f.write(f"MSE: {fastest_convergence['mse']:.6f}\n")
    f.write(f"Training time: {fastest_convergence['total_training_time']:.2f} seconds\n\n")
    
    # Analysis of new hyperparameters
    f.write("=== LEARNING RATE ANALYSIS ===\n")
    for lr in sorted(results['learning_rate'].unique()):
        subset = results[results['learning_rate'] == lr]
        f.write(f"Learning rate {lr}:\n")
        f.write(f"  Count: {len(subset)}\n")
        f.write(f"  Avg MSE: {subset['mse'].mean():.6f}\n")
        f.write(f"  Avg R²: {subset['r2_score'].mean():.6f}\n")
        f.write(f"  Best MSE: {subset['mse'].min():.6f}\n\n")
    
    f.write("=== BATCH SIZE ANALYSIS ===\n")
    for bs in sorted(results['batch_size'].unique()):
        subset = results[results['batch_size'] == bs]
        f.write(f"Batch size {bs}:\n")
        f.write(f"  Count: {len(subset)}\n")
        f.write(f"  Avg MSE: {subset['mse'].mean():.6f}\n")
        f.write(f"  Avg R²: {subset['r2_score'].mean():.6f}\n")
        f.write(f"  Best MSE: {subset['mse'].min():.6f}\n\n")
    
    f.write("=== DROPOUT RATE ANALYSIS ===\n")
    for dr in sorted(results['dropout_rate'].unique()):
        subset = results[results['dropout_rate'] == dr]
        f.write(f"Dropout rate {dr}:\n")
        f.write(f"  Count: {len(subset)}\n")
        f.write(f"  Avg MSE: {subset['mse'].mean():.6f}\n")
        f.write(f"  Avg R²: {subset['r2_score'].mean():.6f}\n")
        f.write(f"  Best MSE: {subset['mse'].min():.6f}\n\n")
    
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
plt.title('MSE by Encoding Dimension and Layer Count (Refined)')
plt.xlabel('Encoding Dimension')
plt.ylabel('Mean Squared Error (lower is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mse_by_encoding_dim_and_layers.pdf'))
plt.close()

# 2. MSE by Learning Rate
plt.figure(figsize=(12, 8))
sns.boxplot(x='learning_rate', y='mse', data=results)
plt.title('MSE by Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Squared Error (lower is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mse_by_learning_rate.pdf'))
plt.close()

# 3. MSE by Batch Size
plt.figure(figsize=(12, 8))
sns.boxplot(x='batch_size', y='mse', data=results)
plt.title('MSE by Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Mean Squared Error (lower is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mse_by_batch_size.pdf'))
plt.close()

# 4. MSE by Dropout Rate
plt.figure(figsize=(12, 8))
sns.boxplot(x='dropout_rate', y='mse', data=results)
plt.title('MSE by Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('Mean Squared Error (lower is better)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mse_by_dropout_rate.pdf'))
plt.close()

# 5. Comparison with Original Results (if available)
if 'original' in combined_results['source'].unique():
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='source', y='mse', data=combined_results)
    plt.title('MSE Comparison: Original vs Refined Tuning')
    plt.xlabel('Source')
    plt.ylabel('Mean Squared Error (lower is better)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mse_comparison_original_vs_refined.pdf'))
    plt.close()
    
    # 6. Top 10 Models Comparison
    top_models = pd.concat([
        original_results.nsmallest(5, 'mse'),
        results.nsmallest(5, 'mse')
    ])
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_models.index, y='mse', hue='source', data=top_models)
    plt.title('Top 10 Models by MSE')
    plt.xlabel('Model Index')
    plt.ylabel('Mean Squared Error (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_models_comparison.pdf'))
    plt.close()

# 7. Heatmap of MSE by Encoding Dimension and Layer Count
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

# 8. Learning Rate vs MSE for Best Configuration
best_config = results[
    (results['encoding_dim'] == 2048) & 
    (results['layer_config'].apply(lambda x: x == [4096]))
]
if len(best_config) > 0:
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='learning_rate', y='mse', hue='batch_size', 
                 style='dropout_rate', markers=True, data=best_config)
    plt.title('Learning Rate vs MSE for Best Configuration (2048, [4096])')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error (lower is better)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_rate_vs_mse_best_config.pdf'))
    plt.close()

print(f"All analysis plots saved to {OUTPUT_DIR}")

# === Create Recommendations ===
recommendations_path = os.path.join(OUTPUT_DIR, "refined_recommendations.txt")
with open(recommendations_path, 'w') as f:
    f.write("===== REFINED POWERTRANSFORMER AUTOENCODER RECOMMENDATIONS =====\n\n")
    
    f.write("Based on the refined hyperparameter tuning results, here are the recommended configurations:\n\n")
    
    f.write("1. BEST OVERALL PERFORMANCE:\n")
    if best_mse['encoding_dim'] == best_r2['encoding_dim'] and str(best_mse['layer_config']) == str(best_r2['layer_config']):
        f.write(f"   Encoding dimension: {best_mse['encoding_dim']}\n")
        f.write(f"   Layer configuration: {best_mse['layer_config']}\n")
        f.write(f"   Learning rate: {best_mse['learning_rate']}\n")
        f.write(f"   Batch size: {best_mse['batch_size']}\n")
        f.write(f"   Dropout rate: {best_mse['dropout_rate']}\n")
        f.write(f"   MSE: {best_mse['mse']:.6f}, R²: {best_mse['r2_score']:.6f}\n")
        f.write(f"   This configuration is optimal for both MSE and R² metrics.\n\n")
    else:
        f.write(f"   Encoding dimension: {best_mse['encoding_dim']}\n")
        f.write(f"   Layer configuration: {best_mse['layer_config']}\n")
        f.write(f"   Learning rate: {best_mse['learning_rate']}\n")
        f.write(f"   Batch size: {best_mse['batch_size']}\n")
        f.write(f"   Dropout rate: {best_mse['dropout_rate']}\n")
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
        f.write(f"   Learning rate: {best_balanced['learning_rate']}\n")
        f.write(f"   Batch size: {best_balanced['batch_size']}\n")
        f.write(f"   Dropout rate: {best_balanced['dropout_rate']}\n")
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
        f.write(f"   Learning rate: {fastest['learning_rate']}\n")
        f.write(f"   Batch size: {fastest['batch_size']}\n")
        f.write(f"   Dropout rate: {fastest['dropout_rate']}\n")
        f.write(f"   MSE: {fastest['mse']:.6f}, Training time: {fastest['total_training_time']:.2f} seconds\n")
        f.write(f"   This configuration prioritizes speed while maintaining acceptable performance.\n\n")
    else:
        f.write("   No configuration found that offers fast training with acceptable performance.\n\n")
    
    f.write("4. HYPERPARAMETER IMPACT ANALYSIS:\n")
    
    # Learning rate impact
    lr_impact = results.groupby('learning_rate')['mse'].mean().sort_values()
    best_lr = lr_impact.index[0]
    worst_lr = lr_impact.index[-1]
    f.write(f"   - Learning rate: {best_lr} tends to give the best average performance.\n")
    f.write(f"   - Learning rate: {worst_lr} tends to give the worst average performance.\n")
    
    # Batch size impact
    bs_impact = results.groupby('batch_size')['mse'].mean().sort_values()
    best_bs = bs_impact.index[0]
    worst_bs = bs_impact.index[-1]
    f.write(f"   - Batch size: {best_bs} tends to give the best average performance.\n")
    f.write(f"   - Batch size: {worst_bs} tends to give the worst average performance.\n")
    
    # Dropout rate impact
    dr_impact = results.groupby('dropout_rate')['mse'].mean().sort_values()
    best_dr = dr_impact.index[0]
    worst_dr = dr_impact.index[-1]
    f.write(f"   - Dropout rate: {best_dr} tends to give the best average performance.\n")
    f.write(f"   - Dropout rate: {worst_dr} tends to give the worst average performance.\n")
    
    # Encoding dimension impact
    enc_dim_mse = results.groupby('encoding_dim')['mse'].mean().sort_values()
    best_enc_dim = enc_dim_mse.index[0]
    worst_enc_dim = enc_dim_mse.index[-1]
    f.write(f"   - Encoding dimension: {best_enc_dim} tends to give the best average performance.\n")
    f.write(f"   - Encoding dimension: {worst_enc_dim} tends to give the worst average performance.\n")
    
    # Layer count impact
    layer_mse = results.groupby('num_layers')['mse'].mean().sort_values()
    best_layers = layer_mse.index[0]
    worst_layers = layer_mse.index[-1]
    f.write(f"   - Models with {best_layers} layers tend to perform best on average.\n")
    f.write(f"   - Models with {worst_layers} layers tend to perform worst on average.\n\n")
    
    # Comparison with original results
    if 'original' in combined_results['source'].unique():
        best_original = original_results.loc[original_results['mse'].idxmin()]
        best_refined = results.loc[results['mse'].idxmin()]
        
        f.write("5. COMPARISON WITH ORIGINAL TUNING:\n")
        f.write(f"   Best original model MSE: {best_original['mse']:.6f}\n")
        f.write(f"   Best refined model MSE: {best_refined['mse']:.6f}\n")
        
        improvement = (best_original['mse'] - best_refined['mse']) / best_original['mse'] * 100
        if improvement > 0:
            f.write(f"   Improvement: {improvement:.2f}%\n\n")
        else:
            f.write(f"   No improvement over original tuning.\n\n")
    
    f.write("6. NEXT STEPS:\n")
    f.write("   - Train the best model on the full dataset and save it for future use.\n")
    f.write("   - Evaluate the model on a separate test set to confirm performance.\n")
    f.write("   - Consider exploring other neural network architectures (e.g., convolutional autoencoders).\n")
    f.write("   - Investigate feature importance by analyzing per-feature reconstruction error.\n")

print(f"Recommendations saved to {recommendations_path}")
print("Analysis complete!")
