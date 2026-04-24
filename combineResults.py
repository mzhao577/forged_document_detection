"""
Combine Binoculars and RoBERTa model results into a single CSV.
Joins on 'filename' and keeps: ai_probability, TrueLabel, model, threshold, Prediction from each.
"""

import sys
import pandas as pd

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <input_file1> <input_file2> <output_file>")
    sys.exit(1)

# Input files from command-line arguments
binocular_file = sys.argv[1]
roberta_file = sys.argv[2]
output_file = sys.argv[3]

# Columns to keep (besides filename)
keep_cols = ["ai_probability", "TrueLabel", "model", "threshold", "Prediction"]

# Load Binoculars results
df_bino = pd.read_csv(binocular_file)
df_bino = df_bino[["filename"] + keep_cols]

# Load RoBERTa results
df_rob = pd.read_csv(roberta_file)
df_rob = df_rob[["filename"] + keep_cols]

# Join on filename with suffixes to distinguish the two models
df_combined = pd.merge(
    df_bino, df_rob,
    on="filename",
    suffixes=("_binocular", "_roberta")
)

# Save
df_combined.to_csv(output_file, index=False)
print(f"Combined {len(df_combined)} rows -> {output_file}")
print(df_combined.head())
