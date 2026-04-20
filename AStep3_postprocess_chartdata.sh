#!/bin/bash
# Batch postprocess text files: apply filtering_cleaning_textFiles.py to each .txt file in the input folder.
#
# Usage: ./postprocess_chartdata.sh <input_folder> <output_folder>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    echo "Example: $0 ./input/raw_texts ./output/cleaned_texts"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/utility3_filtering_cleaning_textFiles.py"


#python filtering_cleaning_textFiles.py input.txt output.txt --min_words 20 --min_chars 80 --min_line_chars 60
minWords=20
minChars=100
minLineCharts=60


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

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count .txt files
FILE_COUNT=$(ls "$INPUT_DIR"/*.txt 2>/dev/null | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "Error: No .txt files found in '$INPUT_DIR'."
    exit 1
fi

echo "Input folder : $INPUT_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "Files to process: $FILE_COUNT"
echo "=========================================="

PROCESSED=0
for INPUT_FILE in "$INPUT_DIR"/*.txt; do
    BASENAME="$(basename "$INPUT_FILE")"
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME"

    echo ""
    echo "[$((PROCESSED + 1))/$FILE_COUNT] Processing: $BASENAME"

    # Filter/clean paragraphs
    python "$PYTHON_SCRIPT" "$INPUT_FILE" "$OUTPUT_FILE" --min_words $minWords  --min_chars $minChars  --min_line_chars $minLineCharts

    PROCESSED=$((PROCESSED + 1))
done

echo ""
echo "=========================================="
echo "Done. Processed $PROCESSED file(s)."
echo "Output saved to: $OUTPUT_DIR"
