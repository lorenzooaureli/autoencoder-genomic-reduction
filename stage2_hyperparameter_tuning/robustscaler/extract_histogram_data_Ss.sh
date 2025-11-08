#!/bin/bash
#SBATCH -A grp-org-sc-mgs
#SBATCH -q jgi_normal
#SBATCH -J extract_histogram
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH --mem=32G
#SBATCH --output=extract_histogram_%j.out
#SBATCH --error=extract_histogram_%j.err

# Load modules
module load python/3.9.12

# Set up environment
export PYTHONPATH=$PYTHONPATH:/global/common/software/m3408/python-packages/cori

# Install required packages if needed
pip install --user tensorflow==2.9.0 keras==2.9.0 h5py==3.7.0 pandas polars matplotlib numpy

# Run the script with the encoding dimension and number of layers as arguments
python /clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/scaled_dataset/tuning_models/robustscaler/extract_histogram_data.py 1024 1

echo "Job completed at $(date)"
