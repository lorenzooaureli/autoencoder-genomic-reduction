#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J autoenc_analysis
#SBATCH -c 10
#SBATCH -t 10:00:00
#SBATCH --mem=32G
#SBATCH --output=autoenc_analysis_robustscaler_%j.out
#SBATCH --error=autoenc_analysis_robustscaler_%j.err

set -euo pipefail

echo "Running autoencoder analysis for RobustScaler"

# Run the Python script with the --no-train flag to use existing model if available
# This avoids the interactive prompt that causes EOFError in batch jobs
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/robustscaler/autoenc_raw_l1_enc2048_loss_graphs_enhanced.py --no-train
