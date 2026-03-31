#!/bin/sh
set -e

MODELS_FILE="/usr/local/bin/models.txt"
MODELS_DIR="/models"

echo "Reading models to download from ${MODELS_FILE}..."

while IFS=' ' read -r repo_id file_name || [ -n "$repo_id" ]; do
    # Skip empty lines or comment lines
    case "$repo_id" in
        ""|\#*) continue ;;
    esac

    target_file="${MODELS_DIR}/${file_name}"
    
    if [ ! -f "$target_file" ]; then
        echo "================================================================"
        echo "Downloading: ${file_name}"
        echo "From Repo:   ${repo_id}"
        echo "(Progress percentage will continuously print below)"
        echo "================================================================"
        
        hf download "$repo_id" "$file_name" \
          --local-dir "$MODELS_DIR"
          
        echo -e "\n[SUCCESS] ${file_name} download complete!"
    else
        echo "================================================================"
        echo "[SKIPPED] ${file_name} already present. Skipping download."
        echo "================================================================"
    fi
done < "$MODELS_FILE"

echo "All operations completed successfully!"