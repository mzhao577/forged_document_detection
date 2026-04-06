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


def write_paragraph_csv(results, output_path):
    """Write paragraph analysis results to CSV."""
    if not results:
        print("No paragraph results to write.")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Classification', 'Confidence', 'AI_Probability', 'Human_Probability', 'Binoculars_Score', 'Word_Count', 'Text'])

        for r in results:
            writer.writerow([
                r['index'],
                r['classification'],
                f"{r['confidence']:.4f}",
                f"{r['ai_probability']:.4f}",
                f"{r['human_probability']:.4f}",
                f"{r['binoculars_score']:.4f}",
                r['word_count'],
                r['text']
            ])

    print(f"Paragraph results written to: {output_path}")


def write_sentence_csv(results, output_path):
    """Write sentence analysis results to CSV."""
    if not results:
        print("No sentence results to write.")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Paragraph_Index', 'Sentence_Index', 'Classification', 'Confidence', 'AI_Probability', 'Human_Probability', 'Binoculars_Score', 'Word_Count', 'Text'])

        for r in results:
            writer.writerow([
                r['index'],
                r['paragraph_index'],
                r['sentence_index'],
                r['classification'],
                f"{r['confidence']:.4f}",
                f"{r['ai_probability']:.4f}",
                f"{r['human_probability']:.4f}",
                f"{r['binoculars_score']:.4f}",
                r['word_count'],
                r['text']
            ])

    print(f"Sentence results written to: {output_path}")


def write_duplicate_csv(pairs, output_path):
    """Write duplicate paragraph pairs to CSV."""
    if not pairs:
        print("No duplicate pairs found.")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Group', 'Paragraph_1_Index', 'Paragraph_2_Index', 'Paragraph_1_Text', 'Paragraph_2_Text'])
        print(f"Empty duplicates file written to: {output_path}")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Paragraph_1_Index', 'Paragraph_2_Index', 'Paragraph_1_Text', 'Paragraph_2_Text'])

        for p in pairs:
            writer.writerow([
                p['group'],
                p['paragraph_1_index'],
                p['paragraph_2_index'],
                p['paragraph_1_text'],
                p['paragraph_2_text']
            ])

    print(f"Duplicate pairs written to: {output_path}")


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


def run_cli_analysis(input_file, output_dir=None, output_prefix="binoculars", model_size="auto",
                     threshold=None, use_api=False, hf_token=None, offline=False):
    """Run CLI analysis and output results to CSV files.

    Args:
        offline: If True, use only locally cached models (no network requests)
    """
    # Read input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    if not text.strip():
        print("Error: Input file is empty.")
        sys.exit(1)

    print(f"Read {len(text)} characters from {input_file}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file) or '.'

    os.makedirs(output_dir, exist_ok=True)

    if use_api:
        # Use Hugging Face Inference API
        observer_client, performer_client = init_api_clients(hf_token)
        actual_model = "falcon (API)"

        # Use Falcon threshold for API mode
        if threshold is None:
            threshold = MODELS["falcon"].get("threshold", DEFAULT_THRESHOLD)
            print(f"Using Falcon threshold: {threshold}")

        # Analyze paragraphs
        print("\nAnalyzing paragraphs via API...")
        para_results = analyze_paragraphs_api(text, observer_client, performer_client, threshold)
        para_output = os.path.join(output_dir, f"{output_prefix}_paragraphs.csv")
        write_paragraph_csv(para_results, para_output)

        # Analyze sentences
        print("\nAnalyzing sentences via API...")
        sent_results = analyze_sentences_api(text, observer_client, performer_client, threshold)
        sent_output = os.path.join(output_dir, f"{output_prefix}_sentences.csv")
        write_sentence_csv(sent_results, sent_output)

        model_description = "Falcon-7B via Hugging Face Inference API"

    else:
        # Use local models
        tokenizer, observer, performer, device, actual_model = load_models_cli(model_size, offline=offline)

        # Use model-specific threshold if not explicitly set
        if threshold is None:
            threshold = MODELS[actual_model].get("threshold", DEFAULT_THRESHOLD)
            print(f"Using model-specific threshold: {threshold}")

        # Analyze paragraphs
        print("\nAnalyzing paragraphs...")
        para_results = analyze_paragraphs_cli(text, tokenizer, observer, performer, device, threshold)
        para_output = os.path.join(output_dir, f"{output_prefix}_paragraphs.csv")
        write_paragraph_csv(para_results, para_output)

        # Analyze sentences
        print("\nAnalyzing sentences...")
        sent_results = analyze_sentences_cli(text, tokenizer, observer, performer, device, threshold)
        sent_output = os.path.join(output_dir, f"{output_prefix}_sentences.csv")
        write_sentence_csv(sent_results, sent_output)

        model_description = MODELS[actual_model]['description']

    # Get duplicate pairs (same for both modes)
    print("\nDetecting duplicate paragraphs...")
    dup_pairs = get_duplicate_pairs(text)
    dup_output = os.path.join(output_dir, f"{output_prefix}_duplicates.csv")
    write_duplicate_csv(dup_pairs, dup_output)

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"  - Paragraphs analyzed: {len(para_results)}")
    print(f"  - Sentences analyzed: {len(sent_results)}")
    print(f"  - Duplicate pairs found: {len(dup_pairs)}")
    print(f"  - Model used: {actual_model} ({model_description})")
    print(f"  - Threshold: {threshold}")
    print("=" * 50)


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
        '--output_result',
        action='store_true',
        help='Enable CLI mode with CSV output'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input text file to analyze (required when --output_result is set)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save CSV output files (default: same as input file directory)'
    )

    parser.add_argument(
        '--output_prefix',
        type=str,
        default='binoculars',
        help='Prefix for output CSV filenames (default: "binoculars")'
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

    if args.output_result:
        if not args.input:
            print("Error: --input is required when using --output_result")
            print("Usage: python binoculars_detection.py --output_result --input <input_file>")
            sys.exit(1)

        run_cli_analysis(
            input_file=args.input,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            model_size=args.model_size,
            threshold=args.threshold,
            use_api=args.use_api,
            hf_token=args.hf_token,
            offline=args.offline
        )
    else:
        print("Usage: python binoculars_detection.py --output_result --input <input_file>")
        print("\nOptions:")
        print("  --output_result    Enable CLI mode with CSV output")
        print("  --input            Input text file to analyze")
        print("  --output_dir       Directory to save CSV output files")
        print("  --output_prefix    Prefix for output CSV filenames (default: 'binoculars')")
        print("  --model_size       Model: auto, falcon, large, medium, small (default: 'auto')")
        print("  --threshold        Detection threshold (default: auto per model, e.g. 0.85 for falcon)")
        print("  --use_api          Use Hugging Face API (no local memory needed, requires HF token)")
        print("  --hf_token         Hugging Face API token (or set HF_TOKEN env var)")
        print("  --offline          Use only locally cached models (no network requests)")
        print("\nExamples:")
        print("  # Local model (auto-selects based on available memory)")
        print("  python binoculars_detection.py --output_result --input sample.txt")
        print("")
        print("  # Use Hugging Face API for Falcon-7B (no local memory needed)")
        print("  python binoculars_detection.py --output_result --input sample.txt --use_api")
        print("")
        print("  # Use locally cached Falcon models (offline mode, no network)")
        print("  python binoculars_detection.py --output_result --input sample.txt --model_size falcon --offline")
