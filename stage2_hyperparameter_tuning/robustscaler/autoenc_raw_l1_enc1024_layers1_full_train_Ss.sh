#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J autoenc_1024_1
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH --mem=120G
#SBATCH --output=autoenc_1024_1_full_train_%j.out
#SBATCH --error=autoenc_1024_1_full_train_%j.err

# -e: Exit the script immediately if any command fails (non-zero exit status).
# -u: Treat unset variables as errors and exit immediately.
# -o pipefail: If any command in a pipeline fails, the entire pipeline fails.
set -euo pipefail

echo "Running autoencoder training and full dataset analysis with best hyperparameters (encoding_dim=1024, num_layers=1, layer_config=[4096])"

# Run the Python script
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/robustscaler/autoenc_raw_l1_enc1024_layers1_full_train.py

echo "Job completed at $(date)"
