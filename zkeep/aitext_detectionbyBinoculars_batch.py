import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import re
import argparse
import csv
import os
import sys
import math

# Optional: Hugging Face Hub for API access
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Model configurations for Binoculars
# The algorithm uses two models: observer (computes perplexity) and performer (baseline)
# Each model has a calibrated threshold since score distributions vary by model size
MODELS = {
    "falcon": {
        "observer": "tiiuae/falcon-7b-instruct",
        "performer": "tiiuae/falcon-7b",
        "description": "Falcon-7B models (~28GB VRAM, best accuracy - original paper)",
        "threshold": 0.85,  # Original paper threshold
        "requires_gpu": True
    },
    "large": {
        "observer": "gpt2-large",
        "performer": "gpt2-medium",
        "description": "GPT-2 Large observer, GPT-2 Medium performer (~4GB, better accuracy)",
        "threshold": 0.92,
        "requires_gpu": False
    },
    "medium": {
        "observer": "gpt2-medium",
        "performer": "gpt2",
        "description": "GPT-2 Medium observer, GPT-2 performer (~2GB, good balance)",
        "threshold": 0.95,  # GPT-2 models produce higher scores than Falcon
        "requires_gpu": False
    },
    "small": {
        "observer": "gpt2",
        "performer": "gpt2",
        "description": "GPT-2 Small (~500MB, fastest, lower accuracy)",
        "threshold": 1.0,  # Small models need higher threshold
        "requires_gpu": False
    }
}

# Fallback order when auto-detecting: try best models first, fall back to smaller ones
FALLBACK_ORDER = ["falcon", "large", "medium", "small"]

# Default model (best accuracy)
DEFAULT_MODEL = "falcon"

# Default detection threshold (used if not specified and no model-specific threshold)
# Binoculars score: lower = more likely AI-generated
DEFAULT_THRESHOLD = 0.85

# Hugging Face API model endpoints
HF_API_MODELS = {
    "observer": "tiiuae/falcon-7b-instruct",
    "performer": "tiiuae/falcon-7b"
}


def get_hf_token():
    """Get Hugging Face token from environment variable."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token


def init_api_clients(hf_token=None):
    """Initialize Hugging Face Inference API clients."""
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub package not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    token = hf_token or get_hf_token()
    if not token:
        print("Error: Hugging Face token required for API access.")
        print("Set HF_TOKEN environment variable or use --hf_token argument.")
        print("Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print("Initializing Hugging Face Inference API clients...")
    observer_client = InferenceClient(model=HF_API_MODELS["observer"], token=token)
    performer_client = InferenceClient(model=HF_API_MODELS["performer"], token=token)

    print(f"  Observer: {HF_API_MODELS['observer']}")
    print(f"  Performer: {HF_API_MODELS['performer']}")

    return observer_client, performer_client


def compute_perplexity_from_logprobs(logprobs):
    """Compute perplexity from a list of log probabilities."""
    if not logprobs:
        return 1.0
    avg_neg_logprob = -sum(logprobs) / len(logprobs)
    return math.exp(avg_neg_logprob)


def get_token_logprobs_api(client, text, max_retries=3):
    """Get token log probabilities from HF Inference API.

    Uses text generation with max_new_tokens=1 to get logprobs for input tokens.
    """
    for attempt in range(max_retries):
        try:
            # Request generation with details to get token logprobs
            response = client.text_generation(
                text,
                max_new_tokens=1,
                details=True,
                return_full_text=False
            )

            # Extract logprobs from prefill tokens (input tokens)
            if hasattr(response, 'details') and response.details:
                prefill = response.details.prefill
                if prefill:
                    logprobs = [token.logprob for token in prefill if token.logprob is not None]
                    return logprobs

            # Fallback: try to get from tokens
            if hasattr(response, 'details') and response.details and response.details.tokens:
                logprobs = [t.logprob for t in response.details.tokens if t.logprob is not None]
                return logprobs

            return []

        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                print(f"  Warning: API call failed after {max_retries} attempts: {e}")
                return []


def compute_binoculars_score_api(text, observer_client, performer_client):
    """
    Compute the Binoculars score using Hugging Face Inference API.

    This is an approximation using log probabilities from the API.
    """
    # Get log probabilities from both models
    observer_logprobs = get_token_logprobs_api(observer_client, text)
    performer_logprobs = get_token_logprobs_api(performer_client, text)

    if not observer_logprobs or not performer_logprobs:
        # Fallback: return neutral score if API fails
        return 0.85

    # Compute perplexities
    observer_ppl = compute_perplexity_from_logprobs(observer_logprobs)
    performer_ppl = compute_perplexity_from_logprobs(performer_logprobs)

    # Binoculars-like score: ratio of log perplexities
    # Lower score = more likely AI generated
    eps = 1e-10
    log_obs_ppl = math.log(observer_ppl + eps)
    log_perf_ppl = math.log(performer_ppl + eps)

    if log_perf_ppl == 0:
        return 0.85  # Neutral fallback

    score = log_obs_ppl / log_perf_ppl

    return float(score)


def get_device():
    """Detect available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_available_memory_gb():
    """Get available GPU/system memory in GB."""
    if torch.cuda.is_available():
        # Get CUDA memory
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / (1024**3)  # Convert to GB
        return total_memory
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) - estimate based on unified memory
        # Can't directly query, so return a conservative estimate
        import subprocess
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            total_bytes = int(result.stdout.strip())
            # Assume about 75% of system memory could be used for GPU
            return (total_bytes / (1024**3)) * 0.75
        except Exception:
            return 8.0  # Conservative default for Apple Silicon
    else:
        # CPU - check system memory
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            total_bytes = int(result.stdout.strip())
            return total_bytes / (1024**3)
        except Exception:
            return 8.0  # Conservative default


def select_best_model(requested_model="auto"):
    """Select the best available model based on system capabilities."""
    if requested_model != "auto" and requested_model in MODELS:
        return requested_model

    available_memory = get_available_memory_gb()
    device = get_device()

    print(f"Auto-detecting best model (available memory: ~{available_memory:.1f}GB, device: {device})...")

    # Memory requirements (approximate)
    memory_requirements = {
        "falcon": 28.0,
        "large": 4.0,
        "medium": 2.0,
        "small": 0.5
    }

    for model_name in FALLBACK_ORDER:
        required = memory_requirements.get(model_name, 2.0)
        if available_memory >= required:
            print(f"  Selected model: {model_name} (requires ~{required}GB)")
            return model_name

    # Fallback to smallest
    print(f"  Limited memory detected, using smallest model: small")
    return "small"


def load_models_cli(model_size="auto", offline=False):
    """Load observer and performer models for Binoculars detection.

    If model_size is "auto", attempts to load the best model for the system,
    starting with Falcon-7B and falling back to smaller models if needed.

    Args:
        model_size: Model size to use ('auto', 'falcon', 'large', 'medium', 'small')
        offline: If True, use only locally cached models (no network requests)
    """
    # Auto-select or validate model
    if model_size == "auto":
        model_size = select_best_model("auto")
    elif model_size not in MODELS:
        print(f"Error: Unknown model size '{model_size}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    model_config = MODELS[model_size]
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading models ({model_config['description']})...")

    try:
        # Load tokenizer (use_fast=True for Rust-based fast tokenizer)
        print(f"  Loading tokenizer from {model_config['observer']}...")
        if offline:
            print("  (Offline mode: using only local cache)")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["observer"],
            local_files_only=offline,
            use_fast=True
        )

        # Set pad token if not present (required for Falcon and some other models)
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            # Also update the model config if needed
            tokenizer.padding_side = "left"

        # Use float16 for MPS (Apple Silicon), bfloat16 for CUDA
        model_dtype = torch.float16 if device == "mps" else torch.bfloat16

        # Load observer model
        print(f"  Loading observer model: {model_config['observer']}...")
        if model_size == "falcon":
            # Check if accelerate is available for device_map="auto"
            try:
                import accelerate
                observer = AutoModelForCausalLM.from_pretrained(
                    model_config["observer"],
                    torch_dtype=model_dtype,
                    device_map="auto",
                    local_files_only=offline
                )
            except ImportError:
                # Fallback: load without device_map and manually move to device
                observer = AutoModelForCausalLM.from_pretrained(
                    model_config["observer"],
                    torch_dtype=model_dtype,
                    local_files_only=offline
                )
                observer = observer.to(device)
        else:
            observer = AutoModelForCausalLM.from_pretrained(model_config["observer"], local_files_only=offline)
            observer = observer.to(device)
        observer.eval()

        # Load performer model
        print(f"  Loading performer model: {model_config['performer']}...")
        if model_size == "falcon":
            try:
                import accelerate
                performer = AutoModelForCausalLM.from_pretrained(
                    model_config["performer"],
                    torch_dtype=model_dtype,
                    device_map="auto",
                    local_files_only=offline
                )
            except ImportError:
                performer = AutoModelForCausalLM.from_pretrained(
                    model_config["performer"],
                    torch_dtype=model_dtype,
                    local_files_only=offline
                )
                performer = performer.to(device)
        else:
            performer = AutoModelForCausalLM.from_pretrained(model_config["performer"], local_files_only=offline)
            performer = performer.to(device)

    except (RuntimeError, torch.cuda.OutOfMemoryError, Exception) as e:
        # If loading fails (e.g., out of memory), try falling back to smaller model
        error_msg = str(e).lower()
        if "memory" in error_msg or "cuda" in error_msg or "mps" in error_msg or "out of" in error_msg or "oom" in error_msg:
            print(f"\n  Warning: Failed to load {model_size} model (memory issue: {e})")

            # Find next smaller model in fallback order
            current_idx = FALLBACK_ORDER.index(model_size) if model_size in FALLBACK_ORDER else 0
            for fallback_model in FALLBACK_ORDER[current_idx + 1:]:
                print(f"  Attempting fallback to: {fallback_model}")
                try:
                    return load_models_cli(fallback_model, offline=offline)
                except Exception:
                    continue

            print("Error: Could not load any model. Please check your system resources.")
            sys.exit(1)
        else:
            raise e
    performer.eval()

    print("Models loaded successfully.")
    return tokenizer, observer, performer, device, model_size


def compute_perplexity(logits, labels, attention_mask):
    """Compute perplexity from logits."""
    # Shift logits and labels for next-token prediction
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()

    # Compute cross-entropy loss per token
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(shifted_logits.transpose(1, 2), shifted_labels)

    # Mask out padding and compute mean
    masked_loss = loss * shifted_mask
    ppl = masked_loss.sum(dim=1) / shifted_mask.sum(dim=1)

    return ppl


def compute_cross_perplexity(observer_logits, performer_logits, attention_mask):
    """Compute cross-perplexity between observer and performer models."""
    # Shift for next-token prediction
    observer_shifted = observer_logits[..., :-1, :].contiguous()
    performer_shifted = performer_logits[..., :-1, :].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()

    # Get performer probabilities
    performer_probs = F.softmax(performer_shifted, dim=-1)

    # Compute cross-entropy: -sum(performer_probs * log_softmax(observer_logits))
    observer_log_probs = F.log_softmax(observer_shifted, dim=-1)
    cross_entropy = -torch.sum(performer_probs * observer_log_probs, dim=-1)

    # Mask and average
    masked_ce = cross_entropy * shifted_mask
    xppl = masked_ce.sum(dim=1) / shifted_mask.sum(dim=1)

    return xppl


def compute_binoculars_score(text, tokenizer, observer, performer, device, max_length=512):
    """
    Compute the Binoculars score for text.

    Score = log(PPL) / log(X-PPL)

    Lower scores indicate AI-generated text.
    """
    # Tokenize (padding=False for single text input to avoid pad token issues)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get logits from both models
    with torch.no_grad():
        observer_outputs = observer(input_ids=input_ids, attention_mask=attention_mask)
        performer_outputs = performer(input_ids=input_ids, attention_mask=attention_mask)

    observer_logits = observer_outputs.logits
    performer_logits = performer_outputs.logits

    # Compute perplexity and cross-perplexity
    ppl = compute_perplexity(observer_logits, input_ids, attention_mask)
    xppl = compute_cross_perplexity(observer_logits, performer_logits, attention_mask)

    # Binoculars score = log(PPL) / log(X-PPL)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_ppl = torch.log(ppl + eps)
    log_xppl = torch.log(xppl + eps)

    score = (log_ppl / log_xppl).cpu().numpy()[0]

    return float(score)


def classify_score(score, threshold=DEFAULT_THRESHOLD):
    """Classify based on Binoculars score."""
    if score < threshold:
        # Lower score = more likely AI
        ai_prob = 1.0 - (score / threshold) if score > 0 else 1.0
        ai_prob = min(max(ai_prob, 0.5), 1.0)  # Clamp between 0.5 and 1.0
        return "AI-written", ai_prob, 1.0 - ai_prob
    else:
        # Higher score = more likely human
        human_prob = min((score - threshold) / (1.0 - threshold) + 0.5, 1.0)
        human_prob = min(max(human_prob, 0.5), 1.0)
        return "Human-written", 1.0 - human_prob, human_prob


def analyze_paragraphs_cli(text, tokenizer, observer, performer, device, threshold=DEFAULT_THRESHOLD):
    """Analyze text paragraph by paragraph with p1, p2, etc. indexing."""
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return []

    results = []
    for i, para in enumerate(paragraphs):
        para_index = f"p{i + 1}"
        word_count = len(para.split())

        # Compute Binoculars score
        score = compute_binoculars_score(para, tokenizer, observer, performer, device)
        classification, ai_prob, human_prob = classify_score(score, threshold)
        confidence = ai_prob if classification == "AI-written" else human_prob

        results.append({
            'index': para_index,
            'text': para,
            'word_count': word_count,
            'classification': classification,
            'confidence': confidence,
            'ai_probability': ai_prob,
            'human_probability': human_prob,
            'binoculars_score': score
        })

    return results


def analyze_sentences_cli(text, tokenizer, observer, performer, device, threshold=DEFAULT_THRESHOLD, min_words=5):
    """Analyze sentences with paragraph+sentence indexing (p1s1, p1s2, etc.)."""
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return []

    results = []

    for para_idx, para in enumerate(paragraphs):
        # Split paragraph into sentences
        sentences = re.split(r'([.!?]+)', para)

        # Reconstruct sentences with punctuation
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sentence:
                reconstructed.append(sentence + punct)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            reconstructed.append(sentences[-1].strip())

        for sent_idx, sentence in enumerate(reconstructed):
            index = f"p{para_idx + 1}s{sent_idx + 1}"
            word_count = len(sentence.split())

            if word_count < min_words:
                results.append({
                    'index': index,
                    'paragraph_index': f"p{para_idx + 1}",
                    'sentence_index': f"s{sent_idx + 1}",
                    'text': sentence,
                    'word_count': word_count,
                    'classification': 'Too short',
                    'confidence': 0,
                    'ai_probability': 0,
                    'human_probability': 0,
                    'binoculars_score': 0
                })
            else:
                score = compute_binoculars_score(sentence, tokenizer, observer, performer, device)
                classification, ai_prob, human_prob = classify_score(score, threshold)
                confidence = ai_prob if classification == "AI-written" else human_prob

                results.append({
                    'index': index,
                    'paragraph_index': f"p{para_idx + 1}",
                    'sentence_index': f"s{sent_idx + 1}",
                    'text': sentence,
                    'word_count': word_count,
                    'classification': classification,
                    'confidence': confidence,
                    'ai_probability': ai_prob,
                    'human_probability': human_prob,
                    'binoculars_score': score
                })

    return results


def detect_duplicates(text):
    """Detect exact duplicate paragraphs in the text."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return None

    paragraph_map = {}

    for i, para in enumerate(paragraphs):
        normalized = ' '.join(para.split()).lower()

        if normalized not in paragraph_map:
            paragraph_map[normalized] = []
        paragraph_map[normalized].append({
            'index': i + 1,
            'text': para
        })

    duplicate_groups = []
    for normalized, paras in paragraph_map.items():
        if len(paras) > 1:
            duplicate_groups.append(paras)

    return {
        'duplicate_groups': duplicate_groups,
        'total_paragraphs': len(paragraphs)
    }


def get_duplicate_pairs(text):
    """Get duplicate paragraph pairs for CSV output."""
    duplicate_result = detect_duplicates(text)

    if not duplicate_result or not duplicate_result['duplicate_groups']:
        return []

    pairs = []
    for group_idx, group in enumerate(duplicate_result['duplicate_groups']):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append({
                    'group': group_idx + 1,
                    'paragraph_1_index': f"p{group[i]['index']}",
                    'paragraph_2_index': f"p{group[j]['index']}",
                    'paragraph_1_text': group[i]['text'],
                    'paragraph_2_text': group[j]['text']
                })

    return pairs


def analyze_paragraphs_api(text, observer_client, performer_client, threshold=DEFAULT_THRESHOLD):
    """Analyze text paragraph by paragraph using HF Inference API."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return []

    results = []
    total = len(paragraphs)
    for i, para in enumerate(paragraphs):
        para_index = f"p{i + 1}"
        word_count = len(para.split())

        print(f"  Processing paragraph {i + 1}/{total}...", end='\r')

        # Compute Binoculars score via API
        score = compute_binoculars_score_api(para, observer_client, performer_client)
        classification, ai_prob, human_prob = classify_score(score, threshold)
        confidence = ai_prob if classification == "AI-written" else human_prob

        results.append({
            'index': para_index,
            'text': para,
            'word_count': word_count,
            'classification': classification,
            'confidence': confidence,
            'ai_probability': ai_prob,
            'human_probability': human_prob,
            'binoculars_score': score
        })

    print()  # New line after progress
    return results


def analyze_sentences_api(text, observer_client, performer_client, threshold=DEFAULT_THRESHOLD, min_words=5):
    """Analyze sentences using HF Inference API."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return []

    results = []
    total_sentences = 0

    # First count total sentences for progress
    for para in paragraphs:
        sentences = re.split(r'([.!?]+)', para)
        total_sentences += (len(sentences) + 1) // 2

    current = 0
    for para_idx, para in enumerate(paragraphs):
        sentences = re.split(r'([.!?]+)', para)

        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sentence:
                reconstructed.append(sentence + punct)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            reconstructed.append(sentences[-1].strip())

        for sent_idx, sentence in enumerate(reconstructed):
            current += 1
            index = f"p{para_idx + 1}s{sent_idx + 1}"
            word_count = len(sentence.split())

            print(f"  Processing sentence {current}/{total_sentences}...", end='\r')

            if word_count < min_words:
                results.append({
                    'index': index,
                    'paragraph_index': f"p{para_idx + 1}",
                    'sentence_index': f"s{sent_idx + 1}",
                    'text': sentence,
                    'word_count': word_count,
                    'classification': 'Too short',
                    'confidence': 0,
                    'ai_probability': 0,
                    'human_probability': 0,
                    'binoculars_score': 0
                })
            else:
                score = compute_binoculars_score_api(sentence, observer_client, performer_client)
                classification, ai_prob, human_prob = classify_score(score, threshold)
                confidence = ai_prob if classification == "AI-written" else human_prob

                results.append({
                    'index': index,
                    'paragraph_index': f"p{para_idx + 1}",
                    'sentence_index': f"s{sent_idx + 1}",
                    'text': sentence,
                    'word_count': word_count,
                    'classification': classification,
                    'confidence': confidence,
                    'ai_probability': ai_prob,
                    'human_probability': human_prob,
                    'binoculars_score': score
                })

    print()  # New line after progress
    return results


def analyze_segments_binoculars(text, tokenizer, observer, performer, device, threshold, segment_words=150):
    """Sliding-window segment analysis mirroring the RoBERTa batch script.

    Splits text into ~segment_words chunks, scores each with Binoculars, and
    reports which segments fall below the threshold (i.e., look AI-generated).
    """
    words = text.split()
    if len(words) < segment_words * 1.5:
        return {
            'is_segmented': False,
            'segments': [],
            'high_ai_segments': [],
            'high_ai_count': 0,
            'total_segments': 0,
            'segment_summary': 'Text too short for segment analysis'
        }

    segments = []
    i = 0
    seg_num = 1
    while i < len(words):
        end = min(i + segment_words, len(words))
        seg_text = ' '.join(words[i:end])
        try:
            score = compute_binoculars_score(seg_text, tokenizer, observer, performer, device)
            _, ai_prob, _ = classify_score(score, threshold)
            segments.append({
                'segment_num': seg_num,
                'word_start': i,
                'word_end': end,
                'binoculars_score': score,
                'ai_probability': ai_prob,
                'is_ai': score < threshold,
                'preview': seg_text[:80] + '...' if len(seg_text) > 80 else seg_text,
            })
        except Exception:
            pass
        i += segment_words
        seg_num += 1
        if seg_num > 15:
            break

    high_ai = [s for s in segments if s['is_ai']]
    if not segments:
        summary = 'Segment analysis failed'
    elif len(high_ai) == 0:
        summary = 'No high-AI segments found'
    elif len(high_ai) == len(segments):
        summary = 'All segments show high AI probability'
    else:
        summary = f"High AI in segments: {[s['segment_num'] for s in high_ai]}"

    return {
        'is_segmented': True,
        'segments': segments,
        'high_ai_segments': high_ai,
        'high_ai_count': len(high_ai),
        'total_segments': len(segments),
        'segment_summary': summary,
    }


def determine_ai_reason_binoculars(score, threshold, ai_prob, dup_pairs, seg_info):
    """Decide a primary reason + contributing factors for an AI classification."""
    reasons = []
    factors = [f"Binoculars score: {score:.4f}", f"Threshold: {threshold}"]

    margin = threshold - score
    if margin > 0.10:
        reasons.append(('Binoculars score far below threshold', 0.95))
    elif margin > 0.03:
        reasons.append(('Binoculars score clearly below threshold', 0.8))
    elif margin > 0:
        reasons.append(('Binoculars score marginally below threshold', 0.6))

    if dup_pairs:
        reasons.append(('Duplicate paragraphs detected', 0.9))
        factors.append(f"Duplicate paragraph pairs: {len(dup_pairs)}")

    if seg_info.get('is_segmented'):
        high = seg_info['high_ai_count']
        total = seg_info['total_segments']
        factors.append(f"High-AI segments: {high}/{total}")
        if total > 1 and high == total:
            reasons.append(('Uniformly high AI across all segments', 0.85))
        elif high > 0:
            reasons.append(('Specific segments show high AI probability', 0.7))

    reasons.sort(key=lambda x: x[1], reverse=True)
    primary = reasons[0][0] if reasons else 'AI patterns detected'
    return primary, factors


def _analyze_one_file(file_path, tokenizer, observer, performer, device, threshold):
    """Analyze a single file and return one summary-row dict (RoBERTa-batch format)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Whole-document Binoculars score
    score = compute_binoculars_score(text, tokenizer, observer, performer, device)
    classification, ai_prob, human_prob = classify_score(score, threshold)
    is_ai = classification == "AI-written"

    # Duplicate detection
    dup_pairs = get_duplicate_pairs(text)
    has_dup = len(dup_pairs) > 0

    # Sliding-window segment analysis (RoBERTa-style, but with Binoculars scoring)
    seg_info = analyze_segments_binoculars(text, tokenizer, observer, performer, device, threshold)
    total_segments = seg_info.get('total_segments', 0)
    high_ai_count = seg_info.get('high_ai_count', 0)

    # duplicate_ratio kept relative to paragraph count for backwards compat
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    total_paras = len(paragraphs)
    dup_ratio = (len(dup_pairs) / total_paras) if total_paras else 0.0

    if is_ai:
        primary_reason, factors = determine_ai_reason_binoculars(score, threshold, ai_prob, dup_pairs, seg_info)
    else:
        primary_reason = ''
        factors = [f"Binoculars score: {score:.4f}", f"Threshold: {threshold}"]
        if seg_info.get('is_segmented'):
            factors.append(f"High-AI segments: {high_ai_count}/{total_segments}")

    return {
        'filename': os.path.basename(file_path),
        'char_count': len(text),
        'word_count': len(text.split()),
        'classification': 'AI_text' if is_ai else 'human_created',
        'ai_probability': f"{ai_prob:.4f}",
        'analysis_reason': primary_reason,
        'has_duplicates': 'Yes' if has_dup else 'No',
        'duplicate_pairs': '; '.join(f"{p['paragraph_1_index']}-{p['paragraph_2_index']}" for p in dup_pairs),
        'duplicate_ratio': f"{dup_ratio:.1%}",
        'high_ai_segments': f"{high_ai_count} of {total_segments}" if seg_info.get('is_segmented') else 'N/A',
        'segment_details': seg_info.get('segment_summary', ''),
        'contributing_factors': '; '.join(factors),
        # Format-compatibility with RoBERTa batch CSV: these columns hold the
        # Binoculars label and Binoculars score respectively.
        'roberta_label': classification,
        'binocular_confidence': f"{score:.4f}",
        'human_probability': f"{human_prob:.4f}",
    }


def run_cli_analysis(input_file, output_dir=None, output_file=None, output_prefix="binoculars",
                     model_size="auto", threshold=None, use_api=False, hf_token=None, offline=False):
    """Analyze a file or folder and write ONE CSV with one row per input file.

    Output columns match aitext_detectionbyRobertA_batch.py exactly.
    """
    import glob as _glob

    if use_api:
        print("Error: --use_api is not supported in single-row-per-file batch mode.")
        sys.exit(1)

    input_path = os.path.expanduser(input_file)

    # Resolve file list (file or folder)
    if os.path.isdir(input_path):
        # Search for .txt files exactly 2 levels deep: inputdir/level2/level3/*.txt
        file_list = sorted(
            _glob.glob(os.path.join(input_path, "*", "*", "*.txt"))
        )
        if not file_list:
            print(f"Error: No .txt files found in '{input_path}' (searched up to 2 levels deep)")
            sys.exit(1)
        is_folder = True
    elif os.path.isfile(input_path):
        file_list = [input_path]
        is_folder = False
    else:
        print(f"Error: '{input_path}' is not a valid file or folder.")
        sys.exit(1)

    # Output dir + filename
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or '.'
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if output_file:
        of = os.path.expanduser(output_file)
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
        output_csv = os.path.join(output_dir, f"{output_prefix}_{base}_results.csv")

    # Load models
    tokenizer, observer, performer, device, actual_model = load_models_cli(model_size, offline=offline)
    if threshold is None:
        threshold = MODELS[actual_model].get("threshold", DEFAULT_THRESHOLD)
        print(f"Using model-specific threshold: {threshold}")

    fieldnames = [
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
        'roberta_label',
        'binocular_confidence',
        'human_probability',
    ]

    print(f"\nProcessing {len(file_list)} file(s)...\n")
    ai_count = 0
    human_count = 0
    error_count = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for i, file_path in enumerate(file_list, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(file_list)}] Processing: {filename}...", end=" ")
            try:
                row = _analyze_one_file(file_path, tokenizer, observer, performer, device, threshold)
                writer.writerow(row)
                if row['classification'] == 'AI_text':
                    ai_count += 1
                else:
                    human_count += 1
                print(f"AI: {row['ai_probability']} - {row['classification']}")
            except Exception as e:
                error_count += 1
                print(f"Error: {e}")

    total = ai_count + human_count
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(file_list)}")
    if total > 0:
        print(f"Detected as AI-generated: {ai_count} ({ai_count/total*100:.1f}%)")
        print(f"Detected as Human-written: {human_count} ({human_count/total*100:.1f}%)")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Model: {actual_model}  Threshold: {threshold}")
    print(f"Results saved to: {output_csv}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Binoculars AI Text Detection - Zero-shot detection of AI-generated text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with auto-detected model (default)
  python binoculars_detection.py --output_result --input mytext.txt

  # Use Hugging Face API for Falcon-7B (no local memory needed)
  python binoculars_detection.py --output_result --input mytext.txt --use_api

  # Use API with explicit token
  python binoculars_detection.py --output_result --input mytext.txt --use_api --hf_token YOUR_TOKEN

  # Force a specific local model
  python binoculars_detection.py --output_result --input mytext.txt --model_size medium

  # Specify output directory and prefix
  python binoculars_detection.py --output_result --input mytext.txt --output_dir ./results --output_prefix myanalysis

  # Use locally cached Falcon models (offline mode, no network requests)
  python binoculars_detection.py --output_result --input mytext.txt --model_size falcon --offline

API Setup:
  1. Create a free account at https://huggingface.co
  2. Get your token at https://huggingface.co/settings/tokens
  3. Set environment variable: export HF_TOKEN=your_token_here
  4. Or pass via --hf_token argument
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input text file or folder of .txt files to analyze'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output CSV filename (or full path). If just a filename, it is written into --output_dir.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save CSV output file (default: same as input directory)'
    )

    parser.add_argument(
        '--output_prefix',
        type=str,
        default='binoculars',
        help='Prefix for default output CSV filename (ignored if --output_file is set)'
    )

    parser.add_argument(
        '--model_size',
        type=str,
        choices=['auto', 'falcon', 'large', 'medium', 'small'],
        default='auto',
        help='Model size: auto (detect best), falcon (~28GB, best), large (~4GB), medium (~2GB), small (~500MB). Default: auto'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Detection threshold (default: auto based on model). Lower scores = more likely AI.'
    )

    parser.add_argument(
        '--use_api',
        action='store_true',
        help='Use Hugging Face Inference API instead of local models. Enables Falcon-7B without local memory requirements.'
    )

    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Hugging Face API token. Can also be set via HF_TOKEN environment variable.'
    )

    parser.add_argument(
        '--offline',
        action='store_true',
        help='Use only locally cached models (no network requests). Models must be pre-downloaded to ~/.cache/huggingface/hub'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_cli_analysis(
        input_file=args.input,
        output_dir=args.output_dir,
        output_file=args.output_file,
        output_prefix=args.output_prefix,
        model_size=args.model_size,
        threshold=args.threshold,
        use_api=args.use_api,
        hf_token=args.hf_token,
        offline=args.offline,
    )
