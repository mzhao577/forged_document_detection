"""
Build a decision tree classifier on combined Binoculars + RoBERTa results.

Usage:
    python CombineModels.py <input_file> <output_file> <output_dir>

Arguments:
    input_file  - combined CSV with ai_probability_binocular, ai_probability_roberta, TrueLabel columns
    output_file - CSV with predictions appended
    output_dir  - directory for confusion matrix and classification report
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
)

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <input_file> <output_file> <output_dir>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
output_dir = sys.argv[3]

os.makedirs(output_dir, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(input_file)

# Use TrueLabel from either side (they should be the same per row)
if "TrueLabel" in df.columns:
    label_col = "TrueLabel"
elif "TrueLabel_binocular" in df.columns:
    label_col = "TrueLabel_binocular"
else:
    print("ERROR: No TrueLabel column found in input file.")
    sys.exit(1)

feature_cols = ["ai_probability_binocular", "ai_probability_roberta"]
X = df[feature_cols].values
y = df[label_col].values

# ---------- Train decision tree ----------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

y_pred = dt.predict(X)
df["DT_Prediction"] = y_pred

# ---------- Extract best split thresholds ----------
tree_rules = export_text(dt, feature_names=feature_cols)
print("=== Decision Tree Rules ===")
print(tree_rules)

# Extract thresholds from the tree structure
tree = dt.tree_
split_info = []
for node_id in range(tree.node_count):
    if tree.feature[node_id] != -2:  # -2 means leaf node
        feat_name = feature_cols[tree.feature[node_id]]
        thresh = tree.threshold[node_id]
        split_info.append({"feature": feat_name, "threshold": round(thresh, 4)})

print("\n=== Split Thresholds ===")
for s in split_info:
    print(f"  {s['feature']}: {s['threshold']}")

# ---------- Confusion matrix ----------
labels = sorted(df[label_col].unique())
cm = confusion_matrix(y, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

print("\n=== Confusion Matrix ===")
print(cm_df)

# ---------- Precision & Recall ----------
pos_label = "AI_Written"
precision = precision_score(y, y_pred, pos_label=pos_label, zero_division=0)
recall = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
report = classification_report(y, y_pred, labels=labels, zero_division=0)

print(f"\nPrecision (AI_Written): {precision:.4f}")
print(f"Recall    (AI_Written): {recall:.4f}")
print(f"\n=== Classification Report ===\n{report}")

# ---------- Save outputs ----------
# 1. Predictions file
df.to_csv(output_file, index=False)
print(f"\nPredictions saved -> {output_file}")

# 2. Confusion matrix CSV
cm_path = os.path.join(output_dir, "confusion_matrix.csv")
cm_df.to_csv(cm_path)
print(f"Confusion matrix  -> {cm_path}")

# 3. Classification report + tree rules
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("=== Decision Tree Rules ===\n")
    f.write(tree_rules + "\n\n")
    f.write("=== Split Thresholds ===\n")
    for s in split_info:
        f.write(f"  {s['feature']}: {s['threshold']}\n")
    f.write(f"\n=== Confusion Matrix ===\n{cm_df}\n")
    f.write(f"\nPrecision (AI_Written): {precision:.4f}\n")
    f.write(f"Recall    (AI_Written): {recall:.4f}\n")
    f.write(f"\n=== Classification Report ===\n{report}\n")
print(f"Classification report -> {report_path}")
