#!/usr/bin/env python3
"""Generate medical charts using OpenAI API from prompt input files.

Usage:
    python generate_charts_HTML.py <input_path> <output_dir>

    input_path:  a single prompt file, or a folder containing multiple prompt files
    output_dir:  directory where output .html and .pdf files will be saved
"""

import sys
import os
import re

# Ensure Homebrew libraries are discoverable by Python/ctypes
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

from openai import OpenAI
from weasyprint import HTML


def generate_chart(input_file: str, output_dir: str):
    """Read a prompt from input_file, send it to OpenAI, save .html and .pdf."""

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
    html_file = os.path.join(output_dir, name_stem + ".html")
    pdf_file = os.path.join(output_dir, name_stem + ".pdf")

    print(f"  Sending prompt to OpenAI...")

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
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
    # - Change fixed positioning to relative so headers/footers don't overlap content
    # - Add proper margins and padding for print layout
    pdf_html = re.sub(
        r'position:\s*fixed',
        'position: relative',
        response_html,
    )
    # Add top margin to <main> so content doesn't sit under the header
    pdf_html = pdf_html.replace(
        '</style>',
        '  main { margin-top: 20px; }\n'
        '  header { border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 10px; }\n'
        '  footer { border-top: 1px solid #ccc; padding-top: 8px; margin-top: 20px; }\n'
        '</style>',
    )

    # Step 3: Convert fixed HTML to PDF using weasyprint
    HTML(string=pdf_html).write_pdf(pdf_file)
    print(f"  PDF  saved to '{pdf_file}'")


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_charts_HTML.py <input_path> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isfile(input_path):
        print(f"Processing '{input_path}'...")
        generate_chart(input_path, output_dir)

    elif os.path.isdir(input_path):
        files = sorted(
            f for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and not f.startswith("~$")
        )
        if not files:
            print(f"No files found in '{input_path}'.")
            sys.exit(1)

        for i, fname in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing '{fname}'...")
            generate_chart(os.path.join(input_path, fname), output_dir)

    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
