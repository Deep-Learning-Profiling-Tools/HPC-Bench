#!/bin/bash

# Define the directories to process
dirs=("EX1" "EX2" "EX3")

# Iterate through each directory
for parent_dir in "${dirs[@]}"; do
    # Check if the directory exists
    if [ -d "$parent_dir" ]; then
        echo "Processing directory: $parent_dir"

        # Iterate through each subdirectory inside the current parent directory
        for subdir in "$parent_dir"/*/; do
            # Ensure it's a directory
            if [ -d "$subdir" ]; then
                optimized_dir="${subdir%/}/${parent_dir}_optimized_codes"
                correctness_dir="${subdir%/}/EX5_correctness"  # Always named EX5_correctness

                # Create directories
                mkdir -p "$optimized_dir"
                mkdir -p "$correctness_dir"

                echo "Created: $optimized_dir"
                echo "Created: $correctness_dir"
            fi
        done
    else
        echo "Skipping: $parent_dir (directory does not exist)"
    fi
done