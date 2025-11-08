#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J autoencoder_tuning
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=powertransformer_tuning_%A_%a.out
#SBATCH --error=powertransformer_tuning_%A_%a.err
#SBATCH --array=1-25

# Load your environment if needed
# module load python/3.9
# source ~/venv/bin/activate

# Create output directories
mkdir -p /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/plots
mkdir -p /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/histograms
mkdir -p /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/summaries
mkdir -p /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/results

# Run script
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/powertransformer/powertransformer_tuning.py
