import argparse


def count_words(text):
    """Count the number of words in text."""
    return len(text.split())


def should_keep_paragraph(paragraph, min_words, min_chars, min_line_chars):
    """
    Determine if a paragraph should be kept based on the filtering rules.

    Returns False (remove) if:
    1. Paragraph has fewer than min_words words OR fewer than min_chars characters
    2. ALL lines in the paragraph have fewer than min_line_chars characters
    """
    lines = paragraph.strip().split('\n')

    # Rule 1: Remove if fewer than min_words words OR fewer than min_chars characters
    word_count = count_words(paragraph)
    char_count = len(paragraph.strip())

    if word_count < min_words or char_count < min_chars:
        return False

    # Rule 2: Remove if ALL lines have fewer than min_line_chars characters
    has_long_line = any(len(line.strip()) >= min_line_chars for line in lines)
    if not has_long_line:
        return False

    return True


def filter_paragraphs(input_path, output_path, min_words, min_chars, min_line_chars):
    """Read input file, filter paragraphs, and write to output file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into paragraphs (separated by blank lines)
    paragraphs = content.split('\n\n')

    # Filter paragraphs
    kept = []
    removed = 0

    for para in paragraphs:
        if para.strip():  # Skip empty paragraphs
            if should_keep_paragraph(para, min_words, min_chars, min_line_chars):
                kept.append(para)
            else:
                removed += 1

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(kept))

    return len(kept), removed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter short paragraphs from a text file'
    )
    parser.add_argument('input_file', help='Input text file to filter')
    parser.add_argument('output_file', help='Output file for filtered text')
    parser.add_argument('--min_words', type=int, default=15,
                        help='Minimum word count for a paragraph (default: 15)')
    parser.add_argument('--min_chars', type=int, default=60,
                        help='Minimum character count for a paragraph (default: 60)')
    parser.add_argument('--min_line_chars', type=int, default=50,
                        help='Minimum character count that at least one line must have (default: 50)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"Filtering paragraphs from: {args.input_file}")
    print(f"Parameters: min_words={args.min_words}, min_chars={args.min_chars}, "
          f"min_line_chars={args.min_line_chars}")

    kept, removed = filter_paragraphs(
        args.input_file, args.output_file,
        args.min_words, args.min_chars, args.min_line_chars
    )

    print(f"\nResults:")
    print(f"  Kept: {kept} paragraphs")
    print(f"  Removed: {removed} paragraphs")
    print(f"  Output saved to: {args.output_file}")
