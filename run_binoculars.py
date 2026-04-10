#!/usr/bin/env python3
"""Simple script to run Binoculars AI text detection on a single text file.

Loads Falcon-7B models from the local HuggingFace cache by default.
"""

import os
import sys
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <text_file>")
        sys.exit(1)

    text_file = sys.argv[1]

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("Error: input file is empty.")
        sys.exit(1)

    print(f"Input file: {text_file}")
    print(f"Text length: {len(text)} chars, {len(text.split())} words\n")

    # Load models from local cache
    tokenizer, observer, performer, device = load_models()

    # Compute score and classify
    score = compute_binoculars_score(text, tokenizer, observer, performer, device)
    classification, ai_prob, human_prob = classify_score(score)

    print(f"{'='*50}")
    print(f"Binoculars Score : {score:.4f}")
    print(f"Threshold        : {DEFAULT_THRESHOLD}")
    print(f"Classification   : {classification}")
    print(f"AI Probability   : {ai_prob:.1%}")
    print(f"Human Probability: {human_prob:.1%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
