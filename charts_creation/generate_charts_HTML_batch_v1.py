#!/usr/bin/env python3
"""Generate N varied medical charts using OpenAI API from prompt input files.

Usage:
    python generate_charts_HTML_batch.py <input_path> <output_dir> <N>

    input_path:  a single prompt file, or a folder containing multiple prompt files
    output_dir:  directory where output .html and .pdf files will be saved
    N:           number of unique variations to generate per prompt
"""

import sys
import os
import re

# Ensure Homebrew libraries are discoverable by Python/ctypes
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

from openai import OpenAI
from weasyprint import HTML

VARIATION_INSTRUCTION = (
    "IMPORTANT: Make this medical record unique and distinct. "
    "Vary ALL of the following from any prior generation: "
    "patient name, age, sex, date of birth, address, phone, MRN, insurance, "
    "provider names and NPIs, facility name and location, "
    "encounter dates, specific diagnoses and findings, "
    "vital signs, lab values, medication names and doses, "
    "clinical narrative style and wording, and charge amounts. "
    "Do NOT reuse any names, dates, or values from previous outputs. "
    "This is variation number {n} of {total} — make it clearly different "
    "from all others."
)


def generate_chart(input_file: str, output_dir: str, variation: int, total: int):
    """Generate one variation of a chart from input_file."""

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, "r") as f:
        prompt = f.read().strip()

    if not prompt:
        print(f"Skipping empty file: '{input_file}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    name_stem = os.path.splitext(os.path.basename(input_file))[0]
    suffix = f"_v{variation:03d}"
    html_file = os.path.join(output_dir, name_stem + suffix + ".html")
    pdf_file = os.path.join(output_dir, name_stem + suffix + ".pdf")

    print(f"  Sending prompt to OpenAI (variation {variation}/{total})...")

    variation_prefix = VARIATION_INSTRUCTION.format(n=variation, total=total)
    full_prompt = variation_prefix + "\n\n" + prompt

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=8192,
        temperature=0.9,
        messages=[{"role": "user", "content": full_prompt}],
    )

    response_html = response.choices[0].message.content

    # Strip markdown code fences if the LLM wraps output in ```html ... ```
    if response_html.startswith("```"):
        lines = response_html.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        response_html = "\n".join(lines)

    # Step 1: Save the original HTML file (as returned by the LLM)
    with open(html_file, "w") as f:
        f.write(response_html)
    print(f"  HTML saved to '{html_file}'")

    # Step 2: Fix CSS for PDF rendering
    pdf_html = re.sub(
        r'position:\s*fixed',
        'position: relative',
        response_html,
    )
    # Remove all page-break rules that create large blank areas
    pdf_html = re.sub(r'page-break-after:\s*always\s*;?', '', pdf_html)
    pdf_html = re.sub(r'page-break-before:\s*always\s*;?', '', pdf_html)
    pdf_html = re.sub(r'page-break-inside:\s*avoid\s*;?', '', pdf_html)
    # Remove blank lines and excessive whitespace between HTML tags
    pdf_html = re.sub(r'\n\s*\n', '\n', pdf_html)
    # Remove empty paragraphs, divs, and <br> runs that create large blanks
    pdf_html = re.sub(r'(<br\s*/?\s*>){3,}', '<br>', pdf_html)
    pdf_html = re.sub(r'<p>\s*</p>', '', pdf_html)
    pdf_html = re.sub(r'<div>\s*</div>', '', pdf_html)
    pdf_html = pdf_html.replace(
        '</style>',
        '  main { margin-top: 20px; }\n'
        '  header { border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 10px; }\n'
        '  footer { border-top: 1px solid #ccc; padding-top: 8px; margin-top: 20px; }\n'
        '  section { padding: 0; margin: 0; }\n'
        '  p { margin: 4px 0; }\n'
        '  h2, h3, h4 { margin: 8px 0 4px 0; }\n'
        '</style>',
    )

    # Step 3: Convert fixed HTML to PDF using weasyprint
    HTML(string=pdf_html).write_pdf(pdf_file)
    print(f"  PDF  saved to '{pdf_file}'")


def process_file(input_file: str, output_dir: str, n: int):
    """Generate N variations for a single prompt file."""
    for v in range(1, n + 1):
        generate_chart(input_file, output_dir, variation=v, total=n)


def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_charts_HTML_batch.py <input_path> <output_dir> <N>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        n = int(sys.argv[3])
        if n < 1:
            raise ValueError
    except ValueError:
        print("Error: N must be a positive integer.")
        sys.exit(1)

    if os.path.isfile(input_path):
        print(f"Processing '{input_path}' — generating {n} variation(s)...")
        process_file(input_path, output_dir, n)

    elif os.path.isdir(input_path):
        files = sorted(
            f for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and not f.startswith("~$")
        )
        if not files:
            print(f"No files found in '{input_path}'.")
            sys.exit(1)

        for i, fname in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing '{fname}' — generating {n} variation(s)...")
            process_file(os.path.join(input_path, fname), output_dir, n)

    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
