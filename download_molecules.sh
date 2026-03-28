#!/usr/bin/env bash
set -euo pipefail

FILE_ID="1IHPr8r0XMVxhBDAtbRDwJ7LaRfI8NxR5"
OUT="dataset.zip"   # change if desired

# Check if gdown is installed
if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Installing it now..."
    pip install --upgrade gdown
fi

echo "Downloading file from Google Drive..."
gdown --id "$FILE_ID" -O "$OUT"

echo "Download complete: $OUT"

