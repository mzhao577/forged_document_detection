#!/bin/bash
# Batch extract text from PDF files: apply utility3_extractTextFromPDF.py to each .pdf file in the input folder.
#
# Usage: ./Step3_extractPDF.sh <input_folder> <output_folder>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    echo "Example: $0 ./input/pdfs ./output/texts"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/utility2_extractTextFromPDF.py"
UNWRAP_SCRIPT="$SCRIPT_DIR/utility4_unwrap_lines.py"

# Validate input folder
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input folder '$INPUT_DIR' does not exist."
    exit 1
fi

# Validate python scripts exist
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi
if [ ! -f "$UNWRAP_SCRIPT" ]; then
    echo "Error: Python script '$UNWRAP_SCRIPT' not found."
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count .pdf files
FILE_COUNT=$(ls "$INPUT_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "Error: No .pdf files found in '$INPUT_DIR'."
    exit 1
fi

echo "Input folder : $INPUT_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "Files to process: $FILE_COUNT"
echo "=========================================="

PROCESSED=0
for INPUT_FILE in "$INPUT_DIR"/*.pdf; do
    BASENAME="$(basename "$INPUT_FILE" .pdf)"
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.txt"

    echo ""
    echo "[$((PROCESSED + 1))/$FILE_COUNT] Processing: $(basename "$INPUT_FILE")"

    python "$PYTHON_SCRIPT" "$INPUT_FILE" "$OUTPUT_FILE"
    python "$UNWRAP_SCRIPT" "$OUTPUT_FILE" "$OUTPUT_FILE"

    PROCESSED=$((PROCESSED + 1))
done

echo ""
echo "=========================================="
echo "Done. Processed $PROCESSED file(s)."
echo "Output saved to: $OUTPUT_DIR"
