#!/bin/bash

# run_visualize_all.sh

# Description:
# This script automates the execution of the 'visualize-all' and 'visualize-adaboost' commands
# for all JSON files within the FINAL_RESULTS directory and its subdirectories.
# It organizes the output plots into a structured directory hierarchy under the specified results directory.

# Usage:
# ./run_visualize_all.sh [RESULTS_BASE_DIR] [FINAL_RESULTS_DIR] [EPOCHS]

# Parameters:
# RESULTS_BASE_DIR   - (Optional) The base directory where visualizations will be saved.
#                      Defaults to 'results'.
# FINAL_RESULTS_DIR  - (Optional) The directory containing JSON result files.
#                      Defaults to 'FINAL_RESULTS'.
# EPOCHS             - (Optional) The epoch number for which to generate confusion matrices.
#                      Defaults to 5.

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 [RESULTS_BASE_DIR] [FINAL_RESULTS_DIR] [EPOCHS]"
    echo ""
    echo "Parameters:"
    echo "  RESULTS_BASE_DIR   - (Optional) The base directory where visualizations will be saved."
    echo "                       Defaults to 'results'."
    echo "  FINAL_RESULTS_DIR  - (Optional) The directory containing JSON result files."
    echo "                       Defaults to 'FINAL_RESULTS'."
    echo "  EPOCHS             - (Optional) The epoch number for which to generate confusion matrices."
    echo "                       Defaults to 5."
    exit 1
}

# Assign input parameters or set default values
RESULTS_BASE_DIR=${1:-"results"}
FINAL_RESULTS_DIR=${2:-"FINAL_RESULTS"}
EPOCHS=${3:-5}

# Validate FINAL_RESULTS_DIR exists
if [ ! -d "$FINAL_RESULTS_DIR" ]; then
    echo "Error: FINAL_RESULTS_DIR '$FINAL_RESULTS_DIR' does not exist."
    usage
fi

# Path to the main.py script
# Assuming run_visualize_all.sh is in the same directory as main.py
MAIN_PY_PATH="./main.py"

# Validate main.py exists
if [ ! -f "$MAIN_PY_PATH" ]; then
    echo "Error: main.py not found at path '$MAIN_PY_PATH'. Please check the script location."
    exit 1
fi

# Collect all JSON files from the specified subdirectories
echo "Collecting JSON files from '$FINAL_RESULTS_DIR'..."

# Initialize arrays to hold JSON paths
JSON_PATHS_ALL=()
JSON_PATHS_ADABOOST=()

# Use find with -print0 to handle spaces and special characters in filenames
while IFS= read -r -d '' file; do
    # Determine the classifier based on the directory structure
    if [[ "$file" == */ADABOOST/* ]]; then
        JSON_PATHS_ADABOOST+=("$file")
    else
        JSON_PATHS_ALL+=("$file")
    fi
done < <(find "$FINAL_RESULTS_DIR" -type f -name "*.json" -print0)

# Check and process non-ADABOOST JSON files
if [ ${#JSON_PATHS_ALL[@]} -gt 0 ]; then
    echo "Found the following JSON files for 'visualize-all':"
    for path in "${JSON_PATHS_ALL[@]}"; do
        echo "$path"
    done
    echo ""

    # Prepare multiple --json-paths flags
    JSON_PATHS_ARGS=()
    for path in "${JSON_PATHS_ALL[@]}"; do
        JSON_PATHS_ARGS+=(--json-paths "$path")
    done

    # Execute the visualize-all command
    echo "Running 'visualize-all' command..."
    python "$MAIN_PY_PATH" visualize-all \
        --results-dir "$RESULTS_BASE_DIR" \
        "${JSON_PATHS_ARGS[@]}" \
        --epochs "$EPOCHS"
    echo ""
else
    echo "No JSON files found for 'visualize-all'. Skipping."
fi

# Check and process ADABOOST JSON files
if [ ${#JSON_PATHS_ADABOOST[@]} -gt 0 ]; then
    echo "Found the following JSON files for 'visualize-adaboost':"
    for path in "${JSON_PATHS_ADABOOST[@]}"; do
        echo "$path"
    done
    echo ""

    # Prepare multiple --json-paths flags
    JSON_PATHS_ADABOOST_ARGS=()
    for path in "${JSON_PATHS_ADABOOST[@]}"; do
        JSON_PATHS_ADABOOST_ARGS+=(--json-paths "$path")
    done

    # Execute the visualize-adaboost command
    echo "Running 'visualize-adaboost' command..."
    python "$MAIN_PY_PATH" visualize-adaboost \
        --results-dir "$RESULTS_BASE_DIR" \
        "${JSON_PATHS_ADABOOST_ARGS[@]}"
    echo ""
else
    echo "No ADABOOST JSON files found. Skipping 'visualize-adaboost'."
fi

echo "All visualizations have been successfully generated and saved to '$RESULTS_BASE_DIR/visualizations/'."