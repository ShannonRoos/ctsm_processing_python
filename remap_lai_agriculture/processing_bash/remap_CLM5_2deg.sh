#!/bin/bash

module load CDO

# Set input and output directories
INPUT_DIR="../masked_5km"
OUTPUT_DIR="../masked_CLM5_2deg"
GRID_FILE="clm5_grid_2deg.txt"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all masked files
for infile in "$INPUT_DIR"/CGLS_*_masked.nc; do
    # Extract base name without path and suffix
    filename=$(basename "$infile" _masked.nc)
    
    # Define output file path
    outfile="$OUTPUT_DIR/${filename}_coarse.nc"

     # Check if output already exists
    if [ -f "$outfile" ]; then
        echo "Skipping $outfile (already exists)"
        continue
    fi
    
    # Perform remapping
    echo "Remapping $infile -> $outfile"
    cdo remapcon,"$GRID_FILE" "$infile" "$outfile"
done

