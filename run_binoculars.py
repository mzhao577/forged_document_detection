#!/usr/bin/env python3
"""Simple script to run Binoculars AI text detection on a single text file.

Supports multiple model families: Falcon-7B (default) and GPT-2 variants (small, medium, large).
Loads models from the local HuggingFace cache.
"""

import os
import sys
import csv
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
    ppl_val = ppl.cpu().numpy()[0]
    xppl_val = xppl.cpu().numpy()[0]
    return float(score), float(ppl_val), float(xppl_val)


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
        description='Binoculars AI Text Detection - Single file mode with CSV output to output directory'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input text file to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for CSV result (default: same directory as input file)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file name (default: <input_basename>_binoculars_result.csv)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=MODEL_CONFIGS.keys(),
                        help='Model to use: falcon (default), small (gpt2), medium (gpt2-medium), large (gpt2-large)')
    return parser.parse_args()


def main():
    args = parse_args()
    text_file = os.path.expanduser(args.input)

    if not os.path.isfile(text_file):
        print(f"Error: '{text_file}' is not a valid file.")
        sys.exit(1)

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("Error: input file is empty.")
        sys.exit(1)

    print(f"Input file: {text_file}")
    print(f"Model     : {args.model}")
    print(f"Text length: {len(text)} chars, {len(text.split())} words\n")

    # Determine output CSV path
    base = os.path.splitext(os.path.basename(text_file))[0]
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    else:
        output_dir = os.path.dirname(text_file) or '.'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = args.output_file if args.output_file else f"{base}_binoculars_result.csv"
    output_csv = os.path.join(output_dir, output_filename)

    # Load models from local cache
    config = MODEL_CONFIGS[args.model]
    tokenizer, observer, performer, device = load_models(config["observer"], config["performer"])

    # Compute score and classify
    score, ppl_val, xppl_val = compute_binoculars_score(text, tokenizer, observer, performer, device)
    classification, ai_prob, human_prob = classify_score(score)

    # Print to console
    print(f"{'='*50}")
    print(f"Model            : {args.model}")
    print(f"Binoculars Score : {score:.4f}")
    print(f"Perplexity       : {ppl_val:.4f}")
    print(f"Cross-Perplexity : {xppl_val:.4f}")
    print(f"Threshold        : {DEFAULT_THRESHOLD}")
    print(f"Classification   : {classification}")
    print(f"AI Probability   : {ai_prob:.1%}")
    print(f"Human Probability: {human_prob:.1%}")
    print(f"{'='*50}")

    # Write CSV output
    fieldnames = ['filename', 'word_count', 'char_count', 'model', 'binoculars_score', 'perplexity', 'cross_perplexity', 'threshold', 'classification', 'ai_probability', 'human_probability']
    row = {
        'filename': os.path.basename(text_file),
        'word_count': len(text.split()),
        'char_count': len(text),
        'model': args.model,
        'binoculars_score': f"{score:.4f}",
        'perplexity': f"{ppl_val:.4f}",
        'cross_perplexity': f"{xppl_val:.4f}",
        'threshold': DEFAULT_THRESHOLD,
        'classification': classification,
        'ai_probability': f"{ai_prob:.4f}",
        'human_probability': f"{human_prob:.4f}",
    }

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()
