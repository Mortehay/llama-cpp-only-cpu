#!/bin/sh
set -e

MODELS_FILE="/usr/local/bin/models.txt"
MODELS_DIR="/models"

# Never echo the token itself — this runs in `make up` and lands in docker logs.
if [ -n "$HF_TOKEN" ]; then echo "HF_TOKEN: set (${#HF_TOKEN} chars)"; else echo "HF_TOKEN: not set — gated repos will fail"; fi

echo "Reading models to download from ${MODELS_FILE}..."

while IFS=' ' read -r repo_id file_name || [ -n "$repo_id" ]; do
    # Strip trailing CR. models.txt is edited on Windows and is checked out with
    # CRLF endings; `read` keeps the \r, which then gets URL-encoded into the
    # download path as %0D and 404s EVERY entry:
    #   Entry Not Found for url: .../Llama-3.2-3B-Instruct-Q4_K_M.gguf%0D
    # .gitattributes now normalises the file, but strip here too so the script
    # works on any checkout regardless of git config.
    repo_id=$(printf '%s' "$repo_id" | tr -d '\r')
    file_name=$(printf '%s' "$file_name" | tr -d '\r')

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
        
        # Omit --token entirely when unset: passing an empty string sends an
        # empty credential, which can fail on ungated repos that need none.
        if [ -n "$HF_TOKEN" ]; then
            hf download "$repo_id" "$file_name" --local-dir "$MODELS_DIR" --token "$HF_TOKEN"
        else
            hf download "$repo_id" "$file_name" --local-dir "$MODELS_DIR"
        fi
          
        echo -e "\n[SUCCESS] ${file_name} download complete!"
    else
        echo "================================================================"
        echo "[SKIPPED] ${file_name} already present. Skipping download."
        echo "================================================================"
    fi
done < "$MODELS_FILE"

echo "All operations completed successfully!"