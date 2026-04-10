#!/usr/bin/env python3
"""Binoculars AI text detection - batch mode.

Processes all .txt files in an input folder (searched up to 2 levels deep)
and writes a single CSV with one row per file.

Usage:
  python aitext_NewBinoculars_batch.py --input /path/to/folder --output_dir /path/to/output
  python aitext_NewBinoculars_batch.py --input /path/to/folder --output_file results.csv
"""

import os
import sys
import glob as _glob
import csv
import re
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local model paths (HuggingFace cache)
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
OBSERVER_PATH = os.path.join(HF_CACHE, "models--tiiuae--falcon-7b-instruct",
                             "snapshots", "8782b5c5d8c9290412416618f36a133653e85285")
PERFORMER_PATH = os.path.join(HF_CACHE, "models--tiiuae--falcon-7b",
                              "snapshots", "ec89142b67d748a1865ea4451372db8313ada0d8")

DEFAULT_THRESHOLD = 0.85


def get_device():
    """Detect available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(observer_path=OBSERVER_PATH, performer_path=PERFORMER_PATH):
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


def detect_duplicate_pairs(text):
    """Detect exact duplicate paragraphs and return pairs."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return [], 0

    paragraph_map = {}
    for i, para in enumerate(paragraphs):
        normalized = ' '.join(para.split()).lower()
        if normalized not in paragraph_map:
            paragraph_map[normalized] = []
        paragraph_map[normalized].append(i + 1)

    pairs = []
    group_idx = 0
    for indices in paragraph_map.values():
        if len(indices) > 1:
            group_idx += 1
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    pairs.append((f"p{indices[a]}", f"p{indices[b]}"))

    return pairs, len(paragraphs)


def analyze_segments(text, tokenizer, observer, performer, device, threshold, segment_words=150):
    """Sliding-window segment analysis. Returns (high_ai_count, total_segments, summary)."""
    words = text.split()
    if len(words) < segment_words * 1.5:
        return 0, 0, "Text too short for segment analysis"

    segments = []
    i = 0
    seg_num = 0
    while i < len(words):
        end = min(i + segment_words, len(words))
        seg_text = ' '.join(words[i:end])
        try:
            score = compute_binoculars_score(seg_text, tokenizer, observer, performer, device)
            segments.append({'num': seg_num + 1, 'is_ai': score < threshold})
        except Exception:
            pass
        i += segment_words
        seg_num += 1
        if seg_num >= 15:
            break

    if not segments:
        return 0, 0, "Segment analysis failed"

    high_ai = [s for s in segments if s['is_ai']]
    high_ai_count = len(high_ai)
    total = len(segments)

    if high_ai_count == 0:
        summary = "No high-AI segments found"
    elif high_ai_count == total:
        summary = "All segments show high AI probability"
    else:
        summary = f"High AI in segments: {[s['num'] for s in high_ai]}"

    return high_ai_count, total, summary


def analyze_one_file(file_path, tokenizer, observer, performer, device, threshold):
    """Analyze a single file and return a dict for the CSV row."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if not text.strip():
        return {
            'filename': os.path.basename(file_path),
            'char_count': 0,
            'word_count': 0,
            'classification': 'error',
            'ai_probability': '',
            'analysis_reason': 'Empty file',
            'has_duplicates': 'No',
            'duplicate_pairs': '',
            'duplicate_ratio': '0.0%',
            'high_ai_segments': 'N/A',
            'segment_details': '',
            'contributing_factors': '',
            'binoculars_label': '',
            'binoculars_score': '',
            'human_probability': '',
        }

    # Whole-document score
    score = compute_binoculars_score(text, tokenizer, observer, performer, device)
    classification, ai_prob, human_prob = classify_score(score, threshold)
    is_ai = classification == "AI-written"

    # Duplicate detection
    dup_pairs, total_paras = detect_duplicate_pairs(text)
    has_dup = len(dup_pairs) > 0
    dup_ratio = (len(dup_pairs) / total_paras) if total_paras else 0.0

    # Segment analysis
    high_ai_count, total_segments, seg_summary = analyze_segments(
        text, tokenizer, observer, performer, device, threshold
    )

    # Build reason and contributing factors
    factors = [f"Binoculars score: {score:.4f}", f"Threshold: {threshold}"]
    primary_reason = ""

    if is_ai:
        margin = threshold - score
        if margin > 0.10:
            primary_reason = "Binoculars score far below threshold"
        elif margin > 0.03:
            primary_reason = "Binoculars score clearly below threshold"
        elif margin > 0:
            primary_reason = "Binoculars score marginally below threshold"
        else:
            primary_reason = "AI patterns detected"

        if has_dup:
            factors.append(f"Duplicate paragraph pairs: {len(dup_pairs)}")
        if total_segments > 0:
            factors.append(f"High-AI segments: {high_ai_count}/{total_segments}")
    else:
        if total_segments > 0:
            factors.append(f"High-AI segments: {high_ai_count}/{total_segments}")

    return {
        'filename': os.path.basename(file_path),
        'char_count': len(text),
        'word_count': len(text.split()),
        'classification': 'AI_text' if is_ai else 'human_created',
        'ai_probability': f"{ai_prob:.4f}",
        'analysis_reason': primary_reason,
        'has_duplicates': 'Yes' if has_dup else 'No',
        'duplicate_pairs': '; '.join(f"{a}-{b}" for a, b in dup_pairs),
        'duplicate_ratio': f"{dup_ratio:.1%}",
        'high_ai_segments': f"{high_ai_count} of {total_segments}" if total_segments > 0 else 'N/A',
        'segment_details': seg_summary,
        'contributing_factors': '; '.join(factors),
        'binoculars_label': classification,
        'binoculars_score': f"{score:.4f}",
        'human_probability': f"{human_prob:.4f}",
    }


CSV_FIELDNAMES = [
    'filename',
    'char_count',
    'word_count',
    'classification',
    'ai_probability',
    'analysis_reason',
    'has_duplicates',
    'duplicate_pairs',
    'duplicate_ratio',
    'high_ai_segments',
    'segment_details',
    'contributing_factors',
    'binoculars_label',
    'binoculars_score',
    'human_probability',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Binoculars AI Text Detection - Batch mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder of .txt files (searched up to 2 levels deep)
  python aitext_NewBinoculars_batch.py --input /path/to/folder --output_dir /path/to/output

  # Specify output filename
  python aitext_NewBinoculars_batch.py --input /path/to/folder --output_file results.csv

  # Process a single file
  python aitext_NewBinoculars_batch.py --input mytext.txt --output_dir ./results

  # Custom threshold
  python aitext_NewBinoculars_batch.py --input /path/to/folder --output_dir ./results --threshold 0.90
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input text file or folder of .txt files to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save CSV output (default: same as input directory)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV filename or full path')
    parser.add_argument('--output_prefix', type=str, default='binoculars',
                        help='Prefix for auto-generated CSV name (default: binoculars)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Detection threshold (default: {DEFAULT_THRESHOLD}). Lower scores = more likely AI.')

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.expanduser(args.input)

    # Resolve file list
    if os.path.isdir(input_path):
        # Search for .txt files exactly 2 levels deep: input/level2/level3/*.txt
        file_list = sorted(_glob.glob(os.path.join(input_path, "*", "*", "*.txt")))
        if not file_list:
            print(f"Error: No .txt files found in '{input_path}' (searched 2 levels deep: input/*//*.txt)")
            sys.exit(1)
        is_folder = True
    elif os.path.isfile(input_path):
        file_list = [input_path]
        is_folder = False
    else:
        print(f"Error: '{input_path}' is not a valid file or folder.")
        sys.exit(1)

    # Determine output CSV path
    if args.output_dir is None:
        output_dir = os.path.dirname(input_path) or '.'
    else:
        output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.output_file:
        of = os.path.expanduser(args.output_file)
        if os.path.isabs(of) or os.path.dirname(of):
            output_csv = of
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
        else:
            output_csv = os.path.join(output_dir, of)
    else:
        if is_folder:
            base = os.path.basename(os.path.normpath(input_path))
        else:
            base = os.path.splitext(os.path.basename(input_path))[0]
        output_csv = os.path.join(output_dir, f"{args.output_prefix}_{base}_results.csv")

    threshold = args.threshold

    # Load models once
    tokenizer, observer, performer, device = load_models()

    print(f"Threshold: {threshold}")
    print(f"Processing {len(file_list)} file(s)...\n")

    ai_count = 0
    human_count = 0
    error_count = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        writer.writeheader()

        for i, file_path in enumerate(file_list, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(file_list)}] Processing: {filename}...", end=" ")
            try:
                row = analyze_one_file(file_path, tokenizer, observer, performer, device, threshold)
                writer.writerow(row)
                if row['classification'] == 'AI_text':
                    ai_count += 1
                elif row['classification'] == 'human_created':
                    human_count += 1
                else:
                    error_count += 1
                print(f"{row['classification']} (score: {row['binoculars_score']})")
            except Exception as e:
                error_count += 1
                print(f"Error: {e}")

    total = ai_count + human_count
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(file_list)}")
    if total > 0:
        print(f"Detected as AI-generated: {ai_count} ({ai_count/total*100:.1f}%)")
        print(f"Detected as Human-written: {human_count} ({human_count/total*100:.1f}%)")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Threshold: {threshold}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
