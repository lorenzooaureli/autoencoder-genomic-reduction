# Stage 4: Production Encoding

Apply trained autoencoder models to encode new datasets.

## Main Tool: autoencoder_tool.py

Unified interface for checking compatibility and encoding datasets.

### Check Compatibility

```bash
python autoencoder_tool.py check <input_file> <scaler_path>
```

Validates:
- Feature count matches
- Metadata columns present
- No missing critical data
- Numeric data types

### Encode Dataset

```bash
python autoencoder_tool.py encode \
  <input_file> \
  <scaler_path> \
  <model_path> \
  [output_file]
```

## Alternative Scripts

- **encode_data.py** - Standalone encoding script
- **check_compatibility.py** - Standalone compatibility checker
- **check_model_encoding_dim.py** - Inspect model architecture
- **encode_final_results.py** - Specific encoding script
- **extract_encoded_features.py** - Feature extraction utility

## Usage Example

```bash
# Step 1: Check compatibility
python autoencoder_tool.py check \
  my_genomic_data.csv \
  robustscaler.pkl

# Step 2: Encode if compatible
python autoencoder_tool.py encode \
  my_genomic_data.csv \
  robustscaler.pkl \
  autoencoder_model.h5 \
  encoded_output.csv
```

## Input Requirements

Your CSV must have:
- Metadata columns: Assembly, Domain, Phylum, Class, Order, Family, Genus, Species
- Label column (if applicable)
- Feature columns matching the number used during training

## Output Format

```csv
Assembly,Domain,...,Label,encoded_0,encoded_1,...,encoded_N
```

Where N = encoding_dim - 1 (e.g., 1023 for encoding_dim=1024)

## Troubleshooting

**"Feature mismatch"**
- Ensure same features as training data
- Check preprocessing steps

**"Out of memory"**
- Model files can be large (1-10 GB)
- Ensure sufficient RAM

**"All zeros in output"**
- Check model/scaler paths
- Verify files aren't corrupted
