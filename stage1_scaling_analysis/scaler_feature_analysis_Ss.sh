#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J feature_scaling
#SBATCH -c 10
#SBATCH -t 10:00:00
#SBATCH --output=scaling_%A_%a.out
#SBATCH --error=scaling_%A_%a.err

set -euo pipefail

# Run Python script
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/scaler_feature_analysis.py

