#!/bin/bash

module load CDO

# Set input and output directories
INPUT_DIR="../esa_cci_1deg"
OUTPUT_DIR="../esa_cci_cgls_1km"
GRID_FILE="../grid_112.txt"

# Make sure output directory exists
#mkdir -p "$OUTPUT_DIR"

# Loop through years 2000 to 2014
for YEAR in {2009..2014}; do
    INPUT_FILE="${INPUT_DIR}/${YEAR}_masked_1deg.nc"
    OUTPUT_FILE="${OUTPUT_DIR}/${YEAR}_masked.nc"
    
    echo "Remapping ${INPUT_FILE} to ${OUTPUT_FILE} ..."
    
    if [ -f "$INPUT_FILE" ]; then
        cdo remapnn,"$GRID_FILE" "$INPUT_FILE" "$OUTPUT_FILE"
    else
        echo "Warning: ${INPUT_FILE} not found, skipping."
    fi
done

echo "All remapping done."
