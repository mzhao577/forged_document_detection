#!/usr/bin/env python3
"""Extract paragraph text from an HTML file and write it to a text file.

Usage:
    python processHTML.py <input.html> <output.txt>
"""

import sys
from bs4 import BeautifulSoup


def extract_paragraphs(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Drop tables, images, and other non-paragraph content so their text
    # doesn't leak into <p> tags via descendants.
    for tag in soup(["table", "img", "figure", "script", "style"]):
        tag.decompose()

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraphs))
        if paragraphs:
            f.write("\n")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python processHTML.py <input.html> <output.txt>", file=sys.stderr)
        sys.exit(1)
    extract_paragraphs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
