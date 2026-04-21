import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/MaxWork/frauddetection/maxBinoculars"))
from binoculars import Binoculars

HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

MODEL_PAIRS = {
    "small": (
        os.path.join(HF_HUB_CACHE, "models--gpt2", "snapshots", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        os.path.join(HF_HUB_CACHE, "models--gpt2-medium", "snapshots", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
    ),
    "large": (
        os.path.join(HF_HUB_CACHE, "models--gpt2-medium", "snapshots", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
        os.path.join(HF_HUB_CACHE, "models--gpt2-large", "snapshots", "32b71b12589c2f8d625668d2335a01cac3249519"),
    ),
    "falcon": (
        os.path.join(HF_HUB_CACHE, "models--tiiuae--falcon-7b", "snapshots", "ec89142b67d748a1865ea4451372db8313ada0d8"),
        os.path.join(HF_HUB_CACHE, "models--tiiuae--falcon-7b-instruct", "snapshots", "8782b5c5d8c9290412416618f36a133653e85285"),
    ),
}

parser = argparse.ArgumentParser(description="Run Binoculars analysis on a folder of text files.")
parser.add_argument("--model", choices=MODEL_PAIRS.keys(), default="small",
                    help="Model pair to use: small, large, or falcon")
parser.add_argument("--input_dir", type=str, required=True,
                    help="Path to folder containing text files")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to store the output CSV file")
parser.add_argument("--output_file", type=str, default="results.csv",
                    help="Output CSV filename (default: results.csv)")
parser.add_argument("--threshold", type=float, default=None,
                    help="Binoculars score threshold for AI detection (default: model's built-in threshold)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, args.output_file)

observer, performer = MODEL_PAIRS[args.model]
print(f"Observer: {observer}, Performer: {performer}")
bino = Binoculars(observer_name_or_path=observer, performer_name_or_path=performer)

text_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))

if not text_files:
    print(f"No .txt files found in {args.input_dir}")
    exit(1)

print(f"Found {len(text_files)} text files\n")

bino_threshold = bino.threshold
threshold = args.threshold if args.threshold is not None else bino_threshold
print(f"Binoculars threshold: {bino_threshold}")
print(f"Using threshold: {threshold}")

with open(output_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "filename", "word_count", "char_count", "model",
        "observer_model", "performer_model",
        "perplexity", "cross_perplexity", "binoculars_score",
        "BinoThreshold", "classification", "ai_probability", "threshold",
        "Prediction", "TrueLabel",
    ])

    for filename in text_files:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"Skipping {filename} (empty file)")
            continue

        word_count = len(text.split())
        char_count = len(text)
        details = bino.compute_score_detailed(text)
        score = details["binoculars_score"]
        prediction = bino.predict(text)
        ai_probability = 1.0 / (1.0 + math.exp(15.0 * (score - threshold)))

        prediction_label = "Human_Written" if score > threshold else "AI_Written"

        if filename.startswith("AI"):
            true_label = "AI_Written"
        elif filename.startswith("Human"):
            true_label = "Human_Written"
        else:
            true_label = ""

        writer.writerow([
            filename,
            word_count,
            char_count,
            args.model,
            details["observer_model"],
            details["performer_model"],
            f"{details['perplexity']:.4f}",
            f"{details['cross_perplexity']:.4f}",
            f"{score:.4f}",
            f"{bino_threshold:.4f}",
            prediction,
            f"{ai_probability:.4f}",
            f"{threshold:.4f}",
            prediction_label,
            true_label,
        ])
        print(f"{filename}: score={score:.4f}, ai_prob={ai_probability:.4f}, {prediction}")

print(f"\nResults saved to {output_path}")
