#!/usr/bin/env python3
"""Batch script to run Binoculars AI text detection on all .txt files in a folder.

Supports multiple model families: Falcon-7B (default) and GPT-2 variants (small, medium, large).
Loads models from the local HuggingFace cache.
"""

import os
import sys
import csv
import glob
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local model paths (HuggingFace cache)
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# Model configurations: each entry maps to (observer_path, performer_path)
MODEL_CONFIGS = {
    "falcon": {
        "observer": os.path.join(HF_CACHE, "models--tiiuae--falcon-7b-instruct",
                                 "snapshots", "8782b5c5d8c9290412416618f36a133653e85285"),
        "performer": os.path.join(HF_CACHE, "models--tiiuae--falcon-7b",
                                  "snapshots", "ec89142b67d748a1865ea4451372db8313ada0d8"),
    },
    "small": {
        "observer": os.path.join(HF_CACHE, "models--gpt2",
                                 "snapshots", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        "performer": os.path.join(HF_CACHE, "models--gpt2",
                                  "snapshots", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    },
    "medium": {
        "observer": os.path.join(HF_CACHE, "models--gpt2-medium",
                                 "snapshots", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
        "performer": os.path.join(HF_CACHE, "models--gpt2",
                                  "snapshots", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    },
    "large": {
        "observer": os.path.join(HF_CACHE, "models--gpt2-large",
                                 "snapshots", "32b71b12589c2f8d625668d2335a01cac3249519"),
        "performer": os.path.join(HF_CACHE, "models--gpt2",
                                  "snapshots", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    },
}

DEFAULT_MODEL = "falcon"
DEFAULT_THRESHOLD = 0.85


def get_device():
    """Detect available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(observer_path, performer_path):
    """Load observer and performer models from local paths."""
    device = get_device()
    model_dtype = torch.float16 if device == "mps" else torch.bfloat16

    print(f"Device: {device}")
    print(f"Loading tokenizer from {observer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(observer_path, local_files_only=True, use_fast=True)

    print(f"Loading observer model: {observer_path}...")
    try:
        import accelerate
        observer = AutoModelForCausalLM.from_pretrained(
            observer_path, torch_dtype=model_dtype, device_map="auto", local_files_only=True
        )
    except ImportError:
        observer = AutoModelForCausalLM.from_pretrained(
            observer_path, torch_dtype=model_dtype, local_files_only=True
        )
        observer = observer.to(device)
    observer.eval()

    print(f"Loading performer model: {performer_path}...")
    try:
        import accelerate
        performer = AutoModelForCausalLM.from_pretrained(
            performer_path, torch_dtype=model_dtype, device_map="auto", local_files_only=True
        )
    except ImportError:
        performer = AutoModelForCausalLM.from_pretrained(
            performer_path, torch_dtype=model_dtype, local_files_only=True
        )
        performer = performer.to(device)
    performer.eval()

    print("Models loaded successfully.\n")
    return tokenizer, observer, performer, device


def compute_binoculars_score(text, tokenizer, observer, performer, device, max_length=512):
    """Compute the Binoculars score. Lower scores indicate AI-generated text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        observer_logits = observer(input_ids=input_ids, attention_mask=attention_mask).logits
        performer_logits = performer(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift for next-token prediction
    shifted_obs = observer_logits[..., :-1, :].contiguous()
    shifted_perf = performer_logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()

    # Perplexity (observer)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    ppl_loss = loss_fn(shifted_obs.transpose(1, 2), shifted_labels)
    ppl = (ppl_loss * shifted_mask).sum(dim=1) / shifted_mask.sum(dim=1)

    # Cross-perplexity (observer vs performer)
    performer_probs = F.softmax(shifted_perf, dim=-1)
    observer_log_probs = F.log_softmax(shifted_obs, dim=-1)
    cross_entropy = -torch.sum(performer_probs * observer_log_probs, dim=-1)
    xppl = (cross_entropy * shifted_mask).sum(dim=1) / shifted_mask.sum(dim=1)

    # Score = log(PPL) / log(X-PPL)
    eps = 1e-10
    score = (torch.log(ppl + eps) / torch.log(xppl + eps)).cpu().numpy()[0]
    return float(score)


def classify_score(score, threshold=DEFAULT_THRESHOLD):
    """Classify based on Binoculars score."""
    if score < threshold:
        ai_prob = 1.0 - (score / threshold) if score > 0 else 1.0
        ai_prob = min(max(ai_prob, 0.5), 1.0)
        return "AI-written", ai_prob, 1.0 - ai_prob
    else:
        human_prob = min((score - threshold) / (1.0 - threshold) + 0.5, 1.0)
        human_prob = min(max(human_prob, 0.5), 1.0)
        return "Human-written", 1.0 - human_prob, human_prob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Binoculars AI Text Detection - Batch mode: process all .txt files in a folder'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input folder containing .txt files to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for CSV result (default: same as input folder)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file name (default: binoculars_batch_result.csv)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=MODEL_CONFIGS.keys(),
                        help='Model to use: falcon (default), small (gpt2), medium (gpt2-medium), large (gpt2-large)')
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = os.path.expanduser(args.input)

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    # Find all .txt files in the folder
    txt_files = sorted(glob.glob(os.path.join(input_folder, "*.txt")))
    if not txt_files:
        print(f"Error: no .txt files found in '{input_folder}'.")
        sys.exit(1)

    print(f"Input folder: {input_folder}")
    print(f"Found {len(txt_files)} .txt file(s)")
    print(f"Model       : {args.model}\n")

    # Determine output CSV path
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    else:
        output_dir = input_folder
    os.makedirs(output_dir, exist_ok=True)
    output_filename = args.output_file if args.output_file else "binoculars_batch_result.csv"
    output_csv = os.path.join(output_dir, output_filename)

    # Load models once
    config = MODEL_CONFIGS[args.model]
    tokenizer, observer, performer, device = load_models(config["observer"], config["performer"])

    # Process each file
    fieldnames = ['filename', 'word_count', 'char_count', 'model', 'binoculars_score', 'threshold', 'classification', 'ai_probability', 'human_probability']
    rows = []

    for i, text_file in enumerate(txt_files, 1):
        basename = os.path.basename(text_file)
        print(f"[{i}/{len(txt_files)}] Processing: {basename}")

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"  Skipping (empty file)")
            continue

        print(f"  Text length: {len(text)} chars, {len(text.split())} words")

        score = compute_binoculars_score(text, tokenizer, observer, performer, device)
        classification, ai_prob, human_prob = classify_score(score)

        print(f"  Score: {score:.4f} -> {classification}")

        rows.append({
            'filename': basename,
            'word_count': len(text.split()),
            'char_count': len(text),
            'model': args.model,
            'binoculars_score': f"{score:.4f}",
            'threshold': DEFAULT_THRESHOLD,
            'classification': classification,
            'ai_probability': f"{ai_prob:.4f}",
            'human_probability': f"{human_prob:.4f}",
        })

    # Write all results to a single CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*50}")
    print(f"Processed {len(rows)} file(s), results saved to: {output_csv}")


if __name__ == "__main__":
    main()
