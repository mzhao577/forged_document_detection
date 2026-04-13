"""
Simple RoBERTa AI Text Detector with Analysis
Usage: python testroberta_On1folder_simple.py <input_file_or_folder> [output_csv] [--model openai|fakespot]

Features:
- AI text detection using RoBERTa models
- Duplicate content detection for AI-classified texts
- Segment-level analysis to identify which portions trigger AI classification
"""

import sys
import os
import csv
import glob
import re
import argparse
from transformers import pipeline

# Model configurations
MODELS = {
    'openai': {
        'name': 'RoBERTa OpenAI Detector',
        'path': '~/.cache/huggingface/hub/models--roberta-base-openai-detector/snapshots/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/'
    },
    'fakespot': {
        'name': 'Fakespot AI Detector',
        'path': '~/.cache/huggingface/hub/models--fakespot-ai--roberta-base-ai-text-detection-v1/snapshots/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/'
    }
}


# =============================================================================
# Analysis Functions for AI-Classified Texts
# =============================================================================

def detect_duplicates(text, min_phrase_length=20, min_occurrences=2):
    """
    Detect duplicate/repeated phrases in text.

    Args:
        text: Input text to analyze
        min_phrase_length: Minimum characters for a phrase to be considered
        min_occurrences: Minimum times a phrase must appear to be flagged

    Returns:
        Dictionary with duplicate analysis results
    """
    # Normalize text
    normalized = ' '.join(text.lower().split())

    # Split into sentences
    sentences = re.split(r'[.!?]+', normalized)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_phrase_length]

    # Find duplicate sentences
    sentence_counts = {}
    for sentence in sentences:
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

    duplicate_sentences = [s for s, count in sentence_counts.items() if count >= min_occurrences]

    # Check for repeated phrases (5-8 word n-grams)
    words = normalized.split()
    phrase_counts = {}

    for n in range(5, 9):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            if len(phrase) >= min_phrase_length:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    repeated_phrases = [p for p, count in phrase_counts.items() if count >= min_occurrences]

    # Combine and dedupe (prefer longer phrases)
    all_duplicates = list(set(duplicate_sentences + repeated_phrases))
    all_duplicates.sort(key=len, reverse=True)

    # Remove phrases that are substrings of longer duplicates
    filtered_duplicates = []
    for phrase in all_duplicates:
        is_substring = any(phrase in longer and phrase != longer for longer in filtered_duplicates)
        if not is_substring:
            filtered_duplicates.append(phrase)

    # Calculate duplicate ratio
    duplicate_chars = sum(
        len(p) * (sentence_counts.get(p, 1) + phrase_counts.get(p, 1) - 1)
        for p in filtered_duplicates
    )
    duplicate_ratio = min(duplicate_chars / len(normalized), 1.0) if normalized else 0.0

    return {
        'has_duplicates': len(filtered_duplicates) > 0,
        'duplicate_count': len(filtered_duplicates),
        'duplicate_ratio': duplicate_ratio,
        'duplicate_phrases': filtered_duplicates[:5]  # Top 5 duplicates
    }


def find_duplicate_paragraph_pairs(text):
    """Return list of (idx_a, idx_b) pairs where paragraph a equals paragraph b (normalized)."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    seen = {}
    pairs = []
    for i, para in enumerate(paragraphs, 1):
        norm = ' '.join(para.split()).lower()
        if norm in seen:
            for j in seen[norm]:
                pairs.append((j, i))
            seen[norm].append(i)
        else:
            seen[norm] = [i]
    return pairs


def analyze_segments(text, classifier, segment_words=150):
    """
    Analyze text in segments to identify which portions have high AI probability.

    Args:
        text: Input text to analyze
        classifier: The loaded classifier pipeline
        segment_words: Approximate words per segment

    Returns:
        Dictionary with segment analysis results
    """
    words = text.split()

    # Only segment if text is long enough
    if len(words) < segment_words * 1.5:
        return {
            'is_segmented': False,
            'segments': [],
            'high_ai_segments': [],
            'segment_summary': 'Text too short for segment analysis'
        }

    segments = []
    i = 0
    segment_num = 1

    while i < len(words):
        end = min(i + segment_words, len(words))
        segment_text = ' '.join(words[i:end])

        try:
            result = classifier(segment_text, truncation=True, max_length=512)
            label = result[0]['label']
            score = result[0]['score']

            # Calculate AI probability for this segment
            if label in ['LABEL_1', 'Fake', 'AI']:
                ai_prob = score
            else:
                ai_prob = 1 - score

            segments.append({
                'segment_num': segment_num,
                'word_start': i,
                'word_end': end,
                'ai_probability': ai_prob,
                'is_ai': ai_prob > 0.5,
                'preview': segment_text[:80] + '...' if len(segment_text) > 80 else segment_text
            })
        except Exception:
            pass

        i += segment_words
        segment_num += 1

        # Limit segments
        if segment_num > 15:
            break

    # Identify high AI segments (>60% probability)
    high_ai_segments = [s for s in segments if s['ai_probability'] > 0.6]

    # Generate summary
    if not segments:
        summary = 'Segment analysis failed'
    elif len(high_ai_segments) == 0:
        summary = 'No high-AI segments found'
    elif len(high_ai_segments) == len(segments):
        summary = 'All segments show high AI probability'
    else:
        high_nums = [s['segment_num'] for s in high_ai_segments]
        summary = f"High AI in segments: {high_nums}"

    return {
        'is_segmented': True,
        'total_segments': len(segments),
        'segments': segments,
        'high_ai_segments': high_ai_segments,
        'high_ai_count': len(high_ai_segments),
        'segment_summary': summary
    }


def determine_ai_reason(ai_prob, duplicate_info, segment_info):
    """
    Determine the primary reason for AI classification.

    Args:
        ai_prob: Overall AI probability
        duplicate_info: Results from detect_duplicates()
        segment_info: Results from analyze_segments()

    Returns:
        Dictionary with reason analysis
    """
    reasons = []
    factors = []

    # Factor 1: Overall probability
    factors.append(f"AI probability: {ai_prob:.1%}")

    if ai_prob > 0.9:
        reasons.append(('Very high overall AI probability', 0.95))
    elif ai_prob > 0.7:
        reasons.append(('High overall AI probability', 0.8))
    elif ai_prob > 0.5:
        reasons.append(('Moderate AI probability', 0.6))

    # Factor 2: Duplicates
    if duplicate_info['has_duplicates']:
        dup_ratio = duplicate_info['duplicate_ratio']
        if dup_ratio > 0.3:
            reasons.append(('High duplicate content detected', 0.9))
            factors.append(f"Duplicate ratio: {dup_ratio:.1%}")
        elif dup_ratio > 0.1:
            reasons.append(('Moderate duplicate content detected', 0.7))
            factors.append(f"Duplicate ratio: {dup_ratio:.1%}")
        else:
            factors.append(f"Minor duplicates: {dup_ratio:.1%}")

    # Factor 3: Segment analysis
    if segment_info.get('is_segmented'):
        high_count = segment_info.get('high_ai_count', 0)
        total = segment_info.get('total_segments', 1)

        if high_count == total and total > 1:
            reasons.append(('Uniformly high AI across all segments', 0.85))
            factors.append('All segments flagged as AI')
        elif high_count > 0:
            reasons.append(('Specific segments show high AI probability', 0.7))
            factors.append(f"High-AI segments: {high_count}/{total}")

    # Sort reasons by confidence and pick primary
    reasons.sort(key=lambda x: x[1], reverse=True)
    primary_reason = reasons[0][0] if reasons else 'AI patterns detected'

    return {
        'primary_reason': primary_reason,
        'factors': factors,
        'all_reasons': [r[0] for r in reasons]
    }


def analyze_file(file_path, classifier, run_detailed_analysis=True):
    """
    Analyze a single file and return results dictionary.

    Args:
        file_path: Path to the text file
        classifier: Loaded classifier pipeline
        run_detailed_analysis: Whether to run detailed analysis for AI-classified texts

    Returns:
        Dictionary with detection results and analysis
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # RoBERTa Classification
    result = classifier(text, truncation=True, max_length=512)
    label = result[0]['label']
    score = result[0]['score']

    # Calculate AI probability
    if label in ['LABEL_1', 'Fake', 'AI']:
        ai_prob = score
        human_prob = 1 - score
    else:
        ai_prob = 1 - score
        human_prob = score

    is_ai = ai_prob > 0.5

    # Base result
    result_dict = {
        'filename': os.path.basename(file_path),
        'char_count': len(text),
        'word_count': len(text.split()),
        'roberta_label': label,
        'roberta_confidence': f"{score:.4f}",
        'ai_probability': f"{ai_prob:.4f}",
        'human_probability': f"{human_prob:.4f}",
        'classification': 'AI_text' if is_ai else 'human_created',
        # Analysis fields (populated for AI_text only)
        'analysis_reason': '',
        'has_duplicates': '',
        'duplicate_pairs': '',
        'duplicate_ratio': '',
        'high_ai_segments': '',
        'segment_details': '',
        'contributing_factors': ''
    }

    # Run detailed analysis for AI-classified texts
    if is_ai and run_detailed_analysis:
        # Detect duplicates
        dup_info = detect_duplicates(text)

        # Analyze segments (for longer texts)
        seg_info = analyze_segments(text, classifier)

        # Determine reason
        reason_info = determine_ai_reason(ai_prob, dup_info, seg_info)

        # Update result with analysis
        result_dict['analysis_reason'] = reason_info['primary_reason']
        para_pairs = find_duplicate_paragraph_pairs(text)
        result_dict['has_duplicates'] = 'Yes' if (dup_info['has_duplicates'] or para_pairs) else 'No'
        result_dict['duplicate_pairs'] = '; '.join(f"p{a}-p{b}" for a, b in para_pairs)
        result_dict['duplicate_ratio'] = f"{dup_info['duplicate_ratio']:.1%}"
        result_dict['high_ai_segments'] = str(seg_info.get('high_ai_count', 0)) if seg_info.get('is_segmented') else 'N/A'
        result_dict['segment_details'] = seg_info.get('segment_summary', '')
        result_dict['contributing_factors'] = '; '.join(reason_info['factors'])

        # Store raw analysis data for printing
        result_dict['_dup_info'] = dup_info
        result_dict['_seg_info'] = seg_info
        result_dict['_reason_info'] = reason_info

    return result_dict


def main():
    parser = argparse.ArgumentParser(description='RoBERTa AI Text Detector')
    parser.add_argument('--input', required=True,
                        help='Input file or folder to analyze')
    parser.add_argument('--output_dir', default='.',
                        help='Directory to write the output CSV into (default: current directory)')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV filename (or full path). If a bare filename, it is written into --output_dir.')
    parser.add_argument('--model', '-m', choices=['openai', 'fakespot'], default='openai',
                        help='Model to use: openai (default) or fakespot')

    args = parser.parse_args()
    input_path = os.path.expanduser(args.input)
    model_choice = args.model

    # Determine if input is file or folder
    if os.path.isdir(input_path):
        # Search for .txt files exactly 2 levels deep: inputdir/level2/level3/*.txt
        file_list = sorted(
            glob.glob(os.path.join(input_path, "*", "*", "*.txt"))
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

    # Default output filename
    if is_folder:
        default_name = f"{os.path.basename(os.path.normpath(input_path))}_results.csv"
    else:
        default_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_results.csv"

    out_dir = os.path.expanduser(args.output_dir)
    if args.output_file:
        of = os.path.expanduser(args.output_file)
        if os.path.isabs(of) or os.path.dirname(of):
            output_csv = of
        else:
            output_csv = os.path.join(out_dir, of)
    else:
        output_csv = os.path.join(out_dir, default_name)

    # Ensure parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(output_csv))
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Get model configuration
    model_config = MODELS[model_choice]

    print(f"{'='*60}")
    print(f"RoBERTa AI Text Detection")
    print(f"{'='*60}")
    print(f"Model: {model_config['name']}")
    print(f"Input: {input_path}")
    print(f"Files to process: {len(file_list)}")
    print(f"Output CSV: {output_csv}")
    print(f"{'='*60}\n")

    # Load RoBERTa model from local path
    print(f"Loading {model_config['name']}...")
    model_path = model_config['path'].replace("~", os.path.expanduser("~"))
    classifier = pipeline("text-classification", model=model_path, local_files_only=True)

    print(f"\nProcessing {len(file_list)} file(s)...\n")

    # CSV fieldnames
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
        'roberta_confidence',
        'human_probability'
    ]

    # Track summary stats
    ai_count = 0
    human_count = 0
    error_count = 0

    # Process files and write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for i, file_path in enumerate(file_list, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(file_list)}] Processing: {filename}...", end=" ")

            try:
                result = analyze_file(file_path, classifier)
                writer.writerow(result)

                if result['classification'] == 'AI_text':
                    ai_count += 1
                    print(f"AI: {result['ai_probability']} - {result['classification']}")

                    # Print analysis details
                    if result.get('analysis_reason'):
                        print(f"           Reason: {result['analysis_reason']}")

                    if result.get('_dup_info') and result['_dup_info']['has_duplicates']:
                        dup_info = result['_dup_info']
                        print(f"           Duplicates: {dup_info['duplicate_ratio']:.1%} of text ({dup_info['duplicate_count']} phrases)")
                        if dup_info['duplicate_phrases']:
                            sample = dup_info['duplicate_phrases'][0][:50]
                            print(f"           Sample: '{sample}...'")

                    if result.get('_seg_info') and result['_seg_info'].get('is_segmented'):
                        seg_info = result['_seg_info']
                        print(f"           Segments: {seg_info['segment_summary']}")
                        # Show high-AI segment previews
                        for seg in seg_info['high_ai_segments'][:2]:
                            print(f"             Seg {seg['segment_num']} ({seg['ai_probability']:.1%}): {seg['preview'][:60]}...")
                else:
                    human_count += 1
                    print(f"AI: {result['ai_probability']} - {result['classification']}")

            except Exception as e:
                error_count += 1
                print(f"Error: {e}")

    # Print summary
    total = ai_count + human_count
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(file_list)}")
    if total > 0:
        print(f"Detected as AI-generated: {ai_count} ({ai_count/total*100:.1f}%)")
        print(f"Detected as Human-written: {human_count} ({human_count/total*100:.1f}%)")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
