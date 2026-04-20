#!/usr/bin/env python3
"""
Script to extract text from a single PDF file.

For image-based PDFs: Uses Tesseract OCR
For text-based PDFs: Uses PyPDF2 for direct text extraction

Usage:
    python utility3_extractTextFromPDF.py input.pdf output.txt
    python utility3_extractTextFromPDF.py input.pdf output.txt --save_tmp --tmpout_dir ./tmp
    python utility3_extractTextFromPDF.py input.pdf output.txt --save_tmp --tmpout_dir ./tmp --type image

Requirements:
    pip install pytesseract pillow pdf2image pypdf2

    Also requires (for image-based PDFs):
    - Tesseract OCR: brew install tesseract (macOS)
    - Poppler (for pdf2image): brew install poppler (macOS)
"""

import argparse
import sys
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    from PyPDF2 import PdfReader
except ImportError as e:
    print(f"Error: Missing required package. {e}")
    print("Install with: pip install pytesseract pillow pdf2image pypdf2")
    sys.exit(1)


def extract_text_from_image_pdf(pdf_path: Path, pages_dir: Path = None, lang: str = "eng") -> str:
    """
    Extract text from an image-based PDF file using Tesseract OCR.
    If pages_dir is provided, per-page results (images and text) are saved there.

    Args:
        pdf_path: Path to the PDF file
        pages_dir: Directory to store per-page images and text files (None to skip saving)
        lang: Language code for OCR (default: 'eng')

    Returns:
        Combined extracted text as a string
    """
    print(f"Converting PDF to images: {pdf_path.name}")
    if pages_dir:
        pages_dir.mkdir(parents=True, exist_ok=True)
        print(f"Per-page results will be saved to: {pages_dir}")
    pages = convert_from_path(pdf_path)

    all_text = []

    for i, page in enumerate(pages, start=1):
        print(f"  Processing page {i}/{len(pages)}...")

        text = pytesseract.image_to_string(page, lang=lang)

        if pages_dir:
            # Save the page image
            image_path = pages_dir / f"page_{i:03d}.png"
            page.save(image_path, "PNG")

            # Save per-page text
            text_path = pages_dir / f"page_{i:03d}.txt"
            text_path.write_text(text, encoding="utf-8")

        all_text.append(f"=== Page {i} ===\n{text}")

    return "\n\n".join(all_text)


def extract_text_from_text_pdf(pdf_path: Path, pages_dir: Path = None) -> str:
    """
    Extract text from a text-based PDF file using PyPDF2.
    If pages_dir is provided, per-page text files are saved there.

    Args:
        pdf_path: Path to the PDF file
        pages_dir: Directory to store per-page text files (None to skip saving)

    Returns:
        Combined extracted text as a string
    """
    print(f"Extracting text from PDF: {pdf_path.name}")
    reader = PdfReader(pdf_path)

    all_text = []

    for i, page in enumerate(reader.pages, start=1):
        print(f"  Processing page {i}/{len(reader.pages)}...")
        text = page.extract_text() or ""

        if pages_dir:
            text_path = pages_dir / f"page_{i:03d}.txt"
            text_path.write_text(text, encoding="utf-8")

        all_text.append(f"=== Page {i} ===\n{text}")

    return "\n\n".join(all_text)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a single PDF file and save to a text file"
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the input PDF file"
    )
    parser.add_argument(
        "output_txt",
        help="Path to the output text file"
    )
    parser.add_argument(
        "--save_tmp",
        action="store_true",
        default=False,
        help="Save temporary per-page results to tmpout_dir (default: off)"
    )
    parser.add_argument(
        "--tmpout_dir",
        default=None,
        help="Directory to store temporary per-page results (a subfolder named after the PDF will be created inside). Required when --save_tmp is used."
    )
    parser.add_argument(
        "--type",
        choices=["text", "image"],
        default="image",
        help="PDF type: 'text' for text-based PDFs, 'image' for scanned/image-based PDFs (default)"
    )

    args = parser.parse_args()

    input_pdf = Path(args.input_pdf)
    output_txt = Path(args.output_txt)

    if not input_pdf.exists():
        print(f"Error: Input PDF not found: {input_pdf}")
        sys.exit(1)

    if not input_pdf.suffix.lower() == ".pdf":
        print(f"Error: Input file is not a PDF: {input_pdf}")
        sys.exit(1)

    if args.save_tmp and not args.tmpout_dir:
        print("Error: --tmpout_dir is required when --save_tmp is used.")
        sys.exit(1)

    # Create output directory if needed
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # Set up per-page output directory if requested
    pages_dir = None
    if args.save_tmp:
        pages_dir = Path(args.tmpout_dir) / input_pdf.stem
        pages_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing as {args.type}-based PDF")
    if pages_dir:
        print(f"Per-page results stored in: {pages_dir}")
    print()

    try:
        if args.type == "image":
            text = extract_text_from_image_pdf(input_pdf, pages_dir)
        else:
            text = extract_text_from_text_pdf(input_pdf, pages_dir)

        output_txt.write_text(text, encoding="utf-8")
        print(f"\nDone! Output saved to: {output_txt}")

    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in PATH.")
        print("Install: brew install tesseract")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing {input_pdf.name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
