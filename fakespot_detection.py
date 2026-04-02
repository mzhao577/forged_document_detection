import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import re
import argparse
import csv
import os
import sys

# Model configurations
MODELS = {
    "Fakespot RoBERTa": {
        "name": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "description": "Fine-tuned for AI text detection by Fakespot",
        "labels": {0: "Human", 1: "AI-Generated"}
    }
}


def load_model_cli(model_key="Fakespot RoBERTa"):
    """Load model and tokenizer for CLI mode (without Streamlit caching)."""
    model_info = MODELS[model_key]
    print(f"Loading model: {model_info['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    model = AutoModelForSequenceClassification.from_pretrained(model_info["name"])
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model


def analyze_paragraphs_cli(text, model, tokenizer):
    """Analyze text paragraph by paragraph with p1, p2, etc. indexing."""
    # Split into paragraphs (by double newlines first, then single newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return []

    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "Human", 1: "AI"}

    results = []
    for i, para in enumerate(paragraphs):
        para_index = f"p{i + 1}"
        word_count = len(para.split())

        # Run prediction
        inputs = tokenizer(
            para,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits.numpy()[0])

        # Determine AI vs Human based on label
        ai_idx = None
        human_idx = None
        for idx, label in id2label.items():
            label_lower = label.lower()
            if any(term in label_lower for term in ["ai", "machine", "fake", "generated"]):
                ai_idx = idx
            else:
                human_idx = idx

        if ai_idx is None:
            ai_idx = 1
        if human_idx is None:
            human_idx = 0

        ai_prob = float(probs[ai_idx])
        human_prob = float(probs[human_idx])
        is_ai = ai_prob > human_prob
        confidence = ai_prob if is_ai else human_prob
        classification = "AI-written" if is_ai else "Human-written"

        results.append({
            'index': para_index,
            'text': para,
            'word_count': word_count,
            'classification': classification,
            'confidence': confidence,
            'ai_probability': ai_prob,
            'human_probability': human_prob
        })

    return results


def analyze_sentences_cli(text, model, tokenizer, min_words=5):
    """Analyze sentences with paragraph+sentence indexing (p1s1, p1s2, etc.)."""
    # First split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return []

    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "Human", 1: "AI"}

    results = []

    for para_idx, para in enumerate(paragraphs):
        # Split paragraph into sentences
        sentences = re.split(r'([.!?]+)', para)

        # Reconstruct sentences with their punctuation
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
                    'human_probability': 0
                })
            else:
                inputs = tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = softmax(outputs.logits.numpy()[0])

                ai_idx = None
                human_idx = None
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if any(term in label_lower for term in ["ai", "machine", "fake", "generated"]):
                        ai_idx = idx
                    else:
                        human_idx = idx

                if ai_idx is None:
                    ai_idx = 1
                if human_idx is None:
                    human_idx = 0

                ai_prob = float(probs[ai_idx])
                human_prob = float(probs[human_idx])
                is_ai = ai_prob > human_prob
                confidence = ai_prob if is_ai else human_prob
                classification = "AI-written" if is_ai else "Human-written"

                results.append({
                    'index': index,
                    'paragraph_index': f"p{para_idx + 1}",
                    'sentence_index': f"s{sent_idx + 1}",
                    'text': sentence,
                    'word_count': word_count,
                    'classification': classification,
                    'confidence': confidence,
                    'ai_probability': ai_prob,
                    'human_probability': human_prob
                })

    return results


def detect_duplicates(text):
    """Detect exact duplicate paragraphs in the text."""
    # Split into paragraphs (by newlines)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return None

    # Track paragraphs by their normalized content
    paragraph_map = {}

    for i, para in enumerate(paragraphs):
        # Normalize: strip whitespace and convert to lowercase for comparison
        normalized = ' '.join(para.split()).lower()

        if normalized not in paragraph_map:
            paragraph_map[normalized] = []
        paragraph_map[normalized].append({
            'index': i + 1,  # 1-based index
            'text': para
        })

    # Find duplicate groups (groups with more than one paragraph)
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
        # Generate all pairs within the group
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
        writer.writerow(['Index', 'Classification', 'Confidence', 'AI_Probability', 'Human_Probability', 'Word_Count', 'Text'])

        for r in results:
            writer.writerow([
                r['index'],
                r['classification'],
                f"{r['confidence']:.4f}",
                f"{r['ai_probability']:.4f}",
                f"{r['human_probability']:.4f}",
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
        writer.writerow(['Index', 'Paragraph_Index', 'Sentence_Index', 'Classification', 'Confidence', 'AI_Probability', 'Human_Probability', 'Word_Count', 'Text'])

        for r in results:
            writer.writerow([
                r['index'],
                r['paragraph_index'],
                r['sentence_index'],
                r['classification'],
                f"{r['confidence']:.4f}",
                f"{r['ai_probability']:.4f}",
                f"{r['human_probability']:.4f}",
                r['word_count'],
                r['text']
            ])

    print(f"Sentence results written to: {output_path}")


def write_duplicate_csv(pairs, output_path):
    """Write duplicate paragraph pairs to CSV."""
    if not pairs:
        print("No duplicate pairs found.")
        # Still create the file with headers
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


def run_cli_analysis(input_file, output_dir=None, output_prefix="analysis"):
    """Run CLI analysis and output results to CSV files."""
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

    # Load model
    tokenizer, model = load_model_cli()

    # Analyze paragraphs
    print("\nAnalyzing paragraphs...")
    para_results = analyze_paragraphs_cli(text, model, tokenizer)
    para_output = os.path.join(output_dir, f"{output_prefix}_paragraphs.csv")
    write_paragraph_csv(para_results, para_output)

    # Analyze sentences
    print("\nAnalyzing sentences...")
    sent_results = analyze_sentences_cli(text, model, tokenizer)
    sent_output = os.path.join(output_dir, f"{output_prefix}_sentences.csv")
    write_sentence_csv(sent_results, sent_output)

    # Get duplicate pairs
    print("\nDetecting duplicate paragraphs...")
    dup_pairs = get_duplicate_pairs(text)
    dup_output = os.path.join(output_dir, f"{output_prefix}_duplicates.csv")
    write_duplicate_csv(dup_pairs, dup_output)

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"  - Paragraphs analyzed: {len(para_results)}")
    print(f"  - Sentences analyzed: {len(sent_results)}")
    print(f"  - Duplicate pairs found: {len(dup_pairs)}")
    print("=" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI Text Detection - Analyze text for AI-generated content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI analysis with CSV output
  python fakespot_detection.py --output_result --input mytext.txt

  # Specify output directory and prefix
  python fakespot_detection.py --output_result --input mytext.txt --output_dir ./results --output_prefix myanalysis
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
        default='analysis',
        help='Prefix for output CSV filenames (default: "analysis")'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.output_result:
        # CLI mode
        if not args.input:
            print("Error: --input is required when using --output_result")
            print("Usage: python fakespot_detection.py --output_result --input <input_file>")
            sys.exit(1)

        run_cli_analysis(
            input_file=args.input,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix
        )
    else:
        print("Usage: python fakespot_detection.py --output_result --input <input_file>")
        print("\nOptions:")
        print("  --output_result    Enable CLI mode with CSV output")
        print("  --input            Input text file to analyze")
        print("  --output_dir       Directory to save CSV output files")
        print("  --output_prefix    Prefix for output CSV filenames (default: 'analysis')")
        print("\nExample:")
        print("  python fakespot_detection.py --output_result --input sample.txt")
