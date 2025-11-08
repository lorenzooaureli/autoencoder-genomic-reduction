#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J autoenc_train
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH --output=autoenc_train_robustscaler_%j.out
#SBATCH --error=autoenc_train_robustscaler_%j.err

# -e: Exit the script immediately if any command fails (non-zero exit status).
# -u: Treat unset variables as errors and exit immediately.
# -o pipefail: If any command in a pipeline fails, the entire pipeline fails.
set -euo pipefail

echo "Running autoencoder analysis for RobustScaler with training"

# Run the Python script with the --train flag to force training a new model
# This avoids the interactive prompt that causes EOFError in batch jobs
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/robustscaler/autoenc_raw_l1_enc2048_loss_graphs_enhanced.py --train
