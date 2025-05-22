#!/bin/bash
# Script to render all .mmd files in docs/diagrams to .png

# This script will be moved to scripts/, diagrams are in docs/diagrams/
# Paths are relative to the project root, from where this script is usually called.
DIAGRAM_SRC_DIR="docs/diagrams"
OUTPUT_DIR="docs/diagrams"

# If this script is in docs/diagrams/:
# DIAGRAM_SRC_DIR="."
# OUTPUT_DIR="."

if [ ! -d "$DIAGRAM_SRC_DIR" ]; then
  echo "Source directory $DIAGRAM_SRC_DIR not found. Make sure you are running this script from the project root."
  exit 1
fi

# Ensure output directory exists (especially if different from source)
mkdir -p "$OUTPUT_DIR"

for mmd_file in "$DIAGRAM_SRC_DIR"/*.mmd; do
  if [ -f "$mmd_file" ]; then
    base_name=$(basename "$mmd_file" .mmd)
    # Output PNG file to the OUTPUT_DIR
    png_file="$OUTPUT_DIR/$base_name.png"
    echo "Rendering $mmd_file to $png_file..."
    
    # Use npx if mmdc is not globally installed, or mmdc directly
    npx @mermaid-js/mermaid-cli -i "$mmd_file" -o "$png_file" -b transparent
    # Example with mmdc if installed globally:
    # mmdc -i "$mmd_file" -o "$png_file"
  fi
done

echo "All diagrams rendered to $OUTPUT_DIR as PNGs."