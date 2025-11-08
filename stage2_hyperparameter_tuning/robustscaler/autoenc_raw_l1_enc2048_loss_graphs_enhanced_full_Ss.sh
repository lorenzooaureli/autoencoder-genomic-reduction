#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J autoenc_full
#SBATCH -c 10
#SBATCH -t 12:00:00
#SBATCH --mem=120G
#SBATCH --output=autoenc_full_analysis_robustscaler_%j.out
#SBATCH --error=autoenc_full_analysis_robustscaler_%j.err

set -euo pipefail

echo "Running full dataset autoencoder analysis for RobustScaler"

# Run the Python script that processes all samples and uses the existing model
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/robustscaler/autoenc_raw_l1_enc2048_loss_graphs_enhanced_full.py
