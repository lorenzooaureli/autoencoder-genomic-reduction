# Autoencoder-Based Genomic Feature Reduction

A comprehensive pipeline for dimensionality reduction of high-dimensional genomic orthogroup data using autoencoders. This workflow compresses tens of thousands of features into optimized low-dimensional representations (64-2048 features) for downstream machine learning tasks.

## Overview

This repository implements a systematic 4-stage workflow:

```
Stage 1: Scaling Analysis → Stage 2: Hyperparameter Tuning → Stage 3: Model Selection → Stage 4: Production Encoding
```

**Use Case:** Genomic orthogroup data with metadata (taxonomy, labels) that needs dimensionality reduction while preserving biological signal.

## Repository Structure

```
autoencoder-genomic-reduction/
├── README.md
├── stage1_scaling_analysis/          # Evaluate scaling methods
│   ├── scaler_feature_analysis.py
│   └── scaler_feature_analysis_Ss.sh
├── stage2_hyperparameter_tuning/     # Grid search autoencoders
│   ├── robustscaler/
│   ├── minmaxscaler/
│   ├── maxabsscaler/
│   ├── powertransformer/
│   ├── standardscaler/
│   └── quantiletransformer/
├── stage3_model_selection/           # Analyze and select best model
│   ├── analyze_tuning_results.py
│   └── analyze_refined_tuning.py
├── stage4_production_encoding/       # Apply model to new data
│   ├── autoencoder_tool.py          # Main tool (recommended)
│   ├── encode_data.py
│   ├── check_compatibility.py
│   ├── check_model_encoding_dim.py
│   ├── encode_final_results.py
│   └── extract_encoded_features.py
└── utilities/                         # Helper scripts
    ├── check_all_scalers.py
    ├── check_dataset.py
    └── check_scaler.py
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/lorenzooaureli/autoencoder-genomic-reduction.git
cd autoencoder-genomic-reduction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Encode New Data (if you have a trained model)

```bash
cd stage4_production_encoding

# Check compatibility
python autoencoder_tool.py check your_data.csv your_scaler.pkl

# Encode dataset
python autoencoder_tool.py encode \
  your_data.csv \
  your_scaler.pkl \
  your_model.h5 \
  output_encoded.csv
```

## Workflow Stages

### Stage 1: Scaling Analysis

Evaluates 6 different scaling methods to see how they affect feature distributions.

**Scalers Tested:**
- StandardScaler
- RobustScaler  
- PowerTransformer
- MaxAbsScaler
- MinMaxScaler
- QuantileTransformer

**Usage:**
```bash
cd stage1_scaling_analysis
python scaler_feature_analysis.py --input your_data.csv
```

**What to look for:**
- Bell-shaped distributions (more Gaussian = better)
- Minimal outliers
- Consistent feature ranges

### Stage 2: Hyperparameter Tuning

Grid search over autoencoder architectures for each scaler.

**Search Grid:**
- **Encoding dimensions:** 64, 128, 256, 1024, 2048
- **Layer depths:** 1-5 hidden layers
- **Total configurations:** 25 per scaler

**Autoencoder Architecture:**
```
Input (n_features)
    ↓
Encoder layers + BatchNorm + Dropout(0.2)
    ↓
Bottleneck (encoding_dim) ← Compressed features extracted here
    ↓
Decoder layers (mirror of encoder)
    ↓
Output (n_features)
```

**Usage:**
```bash
cd stage2_hyperparameter_tuning/robustscaler
python robustscaler_tuning.py

# Or submit SLURM array job (parallelizes 25 configs)
sbatch robustscaler_tuning_Ss.sh
```

**Key Output Metrics:**

| Metric | Description | Ideal |
|--------|-------------|-------|
| `r2_score` | Reconstruction quality | → 1.0 |
| `range_mse` | Scale-independent error | → 0 |
| `encoding_dim` | Compression level | Smaller = more compression |
| `convergence_speed` | Training efficiency | Higher = faster |

### Stage 3: Model Selection

Analyze results and select the best model.

**Selection Criteria:**
1. **High reconstruction quality:** R² > 0.95, low range_mse
2. **Desired compression:** Smaller encoding_dim = more compression
3. **Training efficiency:** Fast convergence, low training time

**Example:**
```python
import pandas as pd

# Load tuning results
df = pd.read_csv('../stage2_hyperparameter_tuning/robustscaler/results/autoencoder_results.csv')

# Find best model with good compression
best = df[(df['encoding_dim'] <= 1024) & (df['r2_score'] > 0.95)].sort_values('encoding_dim')
print(best)
```

### Stage 4: Production Encoding

Apply trained model to new datasets.

**Main Tool:** `autoencoder_tool.py`

```bash
# 1. Check compatibility
python autoencoder_tool.py check input.csv scaler.pkl

# 2. Encode if compatible  
python autoencoder_tool.py encode input.csv scaler.pkl model.h5 output.csv
```

**Process:**
1. Separates metadata from features
2. Applies scaler transformation
3. Extracts bottleneck activations
4. Combines metadata + encoded features
5. Saves to CSV

## Data Format

### Input Format

CSV with metadata + numeric features:

```csv
Assembly,Domain,Phylum,Class,Order,Family,Genus,Species,Label,Feature1,Feature2,...
GCA_001,Bacteria,Proteobacteria,...,Escherichia,coli,pathogen,0.5,1.2,...
```

**Required:**
- Metadata columns: Assembly, Domain, Phylum, Class, Order, Family, Genus, Species
- Label column (if doing classification)
- Numeric feature columns

### Output Format

```csv
Assembly,Domain,...,Label,encoded_0,encoded_1,...,encoded_N
GCA_001,Bacteria,...,pathogen,0.123,0.456,...,0.789
```

Where N = encoding_dim - 1

## Key Concepts

### Why Autoencoders?

- **Non-linear compression:** Captures complex patterns (vs PCA)
- **Preserves information:** High R² = good reconstruction
- **Reduces noise:** Learns robust features
- **Faster downstream ML:** Fewer features = faster training

### Range-MSE Explained

Regular MSE doesn't account for scale:
- Feature [0, 0.01] with MSE=0.0001 → 10% error (bad!)
- Feature [0, 10000] with MSE=0.1 → negligible error (good!)

**Range-MSE** normalizes by feature range, making errors comparable.

### Scaler Selection Guide

| Scaler | Best For |
|--------|----------|
| **RobustScaler** | Data with outliers (genomic counts) |
| **StandardScaler** | Gaussian distributions |
| **PowerTransformer** | Skewed distributions |
| **MinMaxScaler** | Need bounded [0,1] range |
| **MaxAbsScaler** | Sparse data (preserves zeros) |

**For genomic data:** RobustScaler often works best.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib

## Troubleshooting

**"Feature count mismatch"**
- Use same preprocessing as training
- Check for missing/extra columns

**"Out of memory"**
- Models can be large (1-10 GB)
- Process in batches or use smaller model

**"Poor reconstruction quality"**
- Try different scaler
- Increase encoding_dim
- Add more encoder/decoder layers

## Citation

If you use this pipeline in your research, please cite:

```
[Add citation here]
```

## License

MIT license

## Author

Lorenzo Aureli - [@lorenzooaureli](https://github.com/lorenzooaureli)

## Acknowledgments

Developed for genomic orthogroup analysis and taxonomic classification.
