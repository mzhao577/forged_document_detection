import argparse
import re


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


def is_single_address_paragraph(paragraph):
    """
    Filter 4: Remove single-paragraph addresses — a paragraph whose entire
    content is a facility/clinic name with address, with or without phone/fax.

    The paragraph must consist of only one line (no line breaks). Examples:
      "ABC Medical Center, 123 Main St, New York, NY 10001"
      "City Clinic 456 Oak Ave, Suite 200, Chicago, IL 60601 Phone: (312) 555-1234 Fax: (312) 555-5678"
    """
    lines = paragraph.strip().split('\n')
    if len(lines) != 1:
        return False

    line = lines[0].strip()

    # Check for US state abbreviation followed by zip code (e.g. "NY 10001")
    has_state_zip = bool(re.search(
        r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', line
    ))

    # Check for phone or fax pattern (e.g. "(555) 123-4567", "555-123-4567")
    has_phone_fax = bool(re.search(
        r'(?:phone|fax|tel|ph)[:\s]*[\(\d][\d\s\(\)\-\.]{7,}', line, re.IGNORECASE
    )) or bool(re.search(
        r'\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}', line
    ))

    # Check for street address keywords
    has_street = bool(re.search(
        r'\b(?:street|st|avenue|ave|boulevard|blvd|drive|dr|road|rd|lane|ln|way|suite|ste|floor|fl)\b[.,]?',
        line, re.IGNORECASE
    ))

    # Match if it has a state+zip (required for an address) and optionally
    # street keywords and/or phone/fax
    if has_state_zip:
        return True

    # Also match if it has both street and phone/fax patterns
    if has_street and has_phone_fax:
        return True

    return False


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
            if not should_keep_paragraph(para, min_words, min_chars, min_line_chars):
                removed += 1
            elif is_single_address_paragraph(para):
                # Filter 4: Remove single address lines
                removed += 1
            else:
                kept.append(para)

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
