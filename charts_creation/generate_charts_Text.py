#!/usr/bin/env python3
"""Generate medical charts from prompt files using OpenAI API.

Usage:
    python generate_charts.py <input_path> <output_dir>

    input_path:  a single prompt file, or a folder containing multiple prompt files
    output_dir:  directory where output .txt and .pdf files will be saved
"""

import sys
import os
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT


def call_llm(prompt: str) -> str:
    """Send prompt to OpenAI and return the response text."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=8192,
        messages=[
            {
                "role": "system",
                "content": (
                    "Output in plain text only. Do not use markdown formatting "
                    "such as #, ##, **, ```, or markdown tables. Use plain text "
                    "headings, dashes, and spaces for structure and alignment."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def save_text(text: str, filepath: str):
    """Save text to a .txt file."""
    with open(filepath, "w") as f:
        f.write(text)


def save_pdf(text: str, filepath: str):
    """Save text to a .pdf file using reportlab."""
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    style = ParagraphStyle(
        "ChartBody",
        parent=getSampleStyleSheet()["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )

    story = []
    for line in text.split("\n"):
        if not line.strip():
            story.append(Spacer(1, 6))
        else:
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, style))

    doc.build(story)


def process_prompt(input_file: str, output_dir: str):
    """Process a single prompt file: call LLM, save .txt and .pdf outputs."""
    if not os.path.isfile(input_file):
        print(f"Error: '{input_file}' is not a valid file.")
        return

    with open(input_file, "r") as f:
        prompt = f.read().strip()

    if not prompt:
        print(f"Skipping empty file: '{input_file}'")
        return

    name_stem = os.path.splitext(os.path.basename(input_file))[0]
    txt_file = os.path.join(output_dir, name_stem + ".txt")
    pdf_file = os.path.join(output_dir, name_stem + ".pdf")

    print(f"  Sending prompt to OpenAI...")
    response_text = call_llm(prompt)

    save_text(response_text, txt_file)
    print(f"  Text saved to '{txt_file}'")

    save_pdf(response_text, pdf_file)
    print(f"  PDF  saved to '{pdf_file}'")


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_charts.py <input_path> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        # Single file
        print(f"Processing '{input_path}'...")
        process_prompt(input_path, output_dir)

    elif os.path.isdir(input_path):
        # Directory with multiple prompt files
        files = sorted(
            f for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and not f.startswith("~$")
        )
        if not files:
            print(f"No files found in '{input_path}'.")
            sys.exit(1)

        for i, fname in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing '{fname}'...")
            process_prompt(os.path.join(input_path, fname), output_dir)

    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
