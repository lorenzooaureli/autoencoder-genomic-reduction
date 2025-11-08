# Stage 2: Hyperparameter Tuning

This directory contains tuning scripts for different scalers. Each subdirectory runs a grid search over 25 autoencoder configurations.

## Directory Structure

```
stage2_hyperparameter_tuning/
├── robustscaler/
│   ├── robustscaler_tuning.py       # Main tuning script
│   └── robustscaler_tuning_Ss.sh    # SLURM submission script
├── minmaxscaler/
├── maxabsscaler/
├── powertransformer/
├── standardscaler/
└── quantiletransformer/
```

## Hyperparameter Grid

Each script tests:
- **Encoding dimensions:** 64, 128, 256, 1024, 2048
- **Layer configurations:** 1-5 hidden layers
- **Total:** 5 × 5 = 25 models

## Usage

### Single Configuration
```bash
cd robustscaler
python robustscaler_tuning.py
```

### Parallel Execution (SLURM)
```bash
cd robustscaler
sbatch --array=1-25 robustscaler_tuning_Ss.sh
```

The `SLURM_ARRAY_TASK_ID` selects which configuration to run.

## Expected Outputs

After running, each scaler directory will contain:

- `results/autoencoder_results.csv` - Metrics for all configurations
- `plots/` - Training/validation curves
- `histograms/` - Error distributions
- `summaries/` - Text summaries

## Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv('robustscaler/results/autoencoder_results.csv')

# Sort by R² score
print(df.sort_values('r2_score', ascending=False))

# Find best compressed model
best = df[df['encoding_dim'] <= 1024].nlargest(5, 'r2_score')
print(best)
```

## Notes

- Training uses CPU by default (modify for GPU)
- Early stopping prevents overfitting
- Each config takes ~10-30 minutes depending on data size
