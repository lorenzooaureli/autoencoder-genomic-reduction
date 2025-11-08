#!/usr/bin/env python
# Simple script to check the dataset structure

import polars as pl
import sys

# Path to the dataset
data_path = "/clusterfs/jgi/scratch/science/mgs/nelli/lorenzo/ML_models/dataset_16_feb/labeled_df_16_feb.csv"

# Define metadata columns
non_numeric_cols = [
    "Assembly", "Domain", "Phylum", "Class", "Order", "Family",
    "Genus", "Species", "Genome accessions", "Label"
]

# Load only column names first
print(f"Loading schema from {data_path}...")
df_schema = pl.read_csv(data_path, n_rows=5, infer_schema_length=None)

# Print basic info
print(f"Total columns in dataset: {len(df_schema.columns)}")

# Count columns by data type
dtype_counts = {}
for dtype in set(df_schema.dtypes):
    count = sum(1 for dt in df_schema.dtypes if dt == dtype)
    dtype_counts[str(dtype)] = count
print(f"Column counts by data type: {dtype_counts}")

# Check for Orthogroup columns
orthogroup_cols = [col for col in df_schema.columns if "Orthogroup" in col]
print(f"Found {len(orthogroup_cols)} Orthogroup columns")
if len(orthogroup_cols) > 0:
    print(f"First 5 Orthogroup columns: {orthogroup_cols[:5]}")

# Check specifically for Orthogroup138309
if "Orthogroup138309" in df_schema.columns:
    print(f"Orthogroup138309 is present in the dataset")
    col_idx = df_schema.columns.index("Orthogroup138309")
    print(f"Orthogroup138309 data type: {df_schema.dtypes[col_idx]}")
else:
    print(f"Orthogroup138309 is NOT present in the dataset")

# Count numeric columns
numeric_cols = [col for col, dtype in zip(df_schema.columns, df_schema.dtypes)
                if col not in non_numeric_cols and dtype == pl.Float64]
print(f"After filtering non-numeric, column count: {len(numeric_cols)}")

# Exclude Orthogroup138309
numeric_cols_filtered = [col for col in numeric_cols if col != "Orthogroup138309"]
print(f"After excluding Orthogroup138309, column count: {len(numeric_cols_filtered)}")

# Check if Orthogroup138309 was actually excluded
was_excluded = len(numeric_cols) != len(numeric_cols_filtered)
print(f"Was Orthogroup138309 actually excluded? {was_excluded}")

# Save column names to file for reference
with open("column_list.txt", "w") as f:
    f.write(f"Total columns: {len(df_schema.columns)}\n")
    f.write(f"Numeric columns: {len(numeric_cols)}\n")
    f.write(f"Numeric columns after filtering: {len(numeric_cols_filtered)}\n\n")
    
    f.write("Metadata columns:\n")
    for col in non_numeric_cols:
        if col in df_schema.columns:
            f.write(f"{col}\n")
    
    f.write("\nFirst 20 numeric columns:\n")
    for col in numeric_cols[:20]:
        f.write(f"{col}\n")
    
    f.write("\nLast 20 numeric columns:\n")
    for col in numeric_cols[-20:]:
        f.write(f"{col}\n")

print("Done! Check column_list.txt for details.")
