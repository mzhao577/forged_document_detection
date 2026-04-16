#!/usr/bin/env bash
# Extract paragraph text from all HTML files in a directory.
#
# Usage: ./extractHTML.sh <input_dir> <output_dir>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>" >&2
    exit 1
fi

input_dir="$1"
output_dir="$2"

if [ ! -d "$input_dir" ]; then
    echo "Error: input directory '$input_dir' does not exist." >&2
    exit 1
fi

mkdir -p "$output_dir"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

count=0
for html_file in "$input_dir"/*.html; do
    [ -e "$html_file" ] || continue

    basename="$(basename "$html_file" .html)"
    output_file="$output_dir/${basename}.txt"

    echo "Processing: $html_file -> $output_file"
    python3 "$SCRIPT_DIR/utility1_extractTextFromhtml.py" "$html_file" "$output_file"
    count=$((count + 1))
done

echo "Done. Processed $count HTML file(s)."
