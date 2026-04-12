#!/usr/bin/env python3
"""Unwrap text files where lines were broken at a fixed character width.

Joins continuation lines within each paragraph into a single long line.
Blank lines (paragraph separators) are preserved.

Usage: python unwrap_lines.py <input_file> <output_file>
"""

import argparse


def unwrap_text(input_path, output_path):
    """Read input file, join wrapped lines within paragraphs, write output."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    paragraphs = []
    current = []

    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.strip() == '':
            # Blank line = paragraph boundary
            if current:
                paragraphs.append(' '.join(current))
                current = []
            paragraphs.append('')  # preserve blank line
        else:
            current.append(stripped.strip())

    # Flush last paragraph
    if current:
        paragraphs.append(' '.join(current))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(paragraphs) + '\n')

    # Count non-empty paragraphs
    para_count = sum(1 for p in paragraphs if p.strip())
    return para_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unwrap lines that were broken at a fixed character width'
    )
    parser.add_argument('input_file', help='Input text file with wrapped lines')
    parser.add_argument('output_file', help='Output file with unwrapped lines')
    args = parser.parse_args()

    count = unwrap_text(args.input_file, args.output_file)
    print(f"Unwrapped {count} paragraph(s): {args.input_file} -> {args.output_file}")
