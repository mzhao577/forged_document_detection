import sys
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python ResultsAnalysis.py <input_file> <output_dir>")
    sys.exit(1)

input_file = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

print(f"Columns found: {list(df.columns)}\n")

# Print label distributions
print("TrueLabel distribution:")
print(df["TrueLabel"].value_counts())
print("\nPrediction distribution:")
print(df["Prediction"].value_counts())

# Confusion matrix
labels = ["AI_Written", "Human_Written"]
cm = confusion_matrix(df["TrueLabel"], df["Prediction"], labels=labels)

print("\nConfusion Matrix:")
print(f"{'':>20} {'Predicted':>25}")
print(f"{'':>20} {'AI_Written':>12} {'Human_Written':>13}")
print(f"{'Actual AI_Written':>20} {cm[0][0]:>12} {cm[0][1]:>13}")
print(f"{'Actual Human_Written':>20} {cm[1][0]:>12} {cm[1][1]:>13}")

# Classification report
report_text = classification_report(df["TrueLabel"], df["Prediction"], labels=labels)
print("\nClassification Report:")
print(report_text)

# 1) Save confusion matrix
cm_path = os.path.join(output_dir, "confusion_matrix.csv")
cm_df = pd.DataFrame(cm, index=["Actual_AI_Written", "Actual_Human_Written"],
                     columns=["Predicted_AI_Written", "Predicted_Human_Written"])
cm_df.to_csv(cm_path)
print(f"Confusion matrix saved to {cm_path}")

# 2) Save precision and recall report
report_path = os.path.join(output_dir, "precision_recall_report.csv")
report_dict = classification_report(df["TrueLabel"], df["Prediction"],
                                    labels=labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(report_path)
print(f"Precision/recall report saved to {report_path}")

# 3) Save confusion matrix chart
chart_path = os.path.join(output_dir, "confusion_matrix_chart.png")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: TrueLabel vs Prediction")
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
print(f"Confusion matrix chart saved to {chart_path}")

# Show chart on screen briefly then exit
plt.show(block=False)
plt.pause(3)
plt.close('all')
