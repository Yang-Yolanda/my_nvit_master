#!/bin/bash

# 3DPW Dataset Download Script
# Downloads the complete 3DPW dataset from official source

set -e

echo "📥 3DPW Dataset Downloader"
echo "=========================="
echo ""

# Configuration
DATA_DIR="/home/yangz/4D-Humans/data/3DPW"
TEMP_DIR="/tmp/3dpw_download"
BASE_URL="https://virtualhumans.mpi-inf.mpg.de/3DPW"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TEMP_DIR"

echo "📂 Target directory: $DATA_DIR"
echo "🌐 Source: $BASE_URL"
echo ""

# Function to download and extract
download_and_extract() {
    local filename=$1
    local url="$BASE_URL/$filename"
    local temp_file="$TEMP_DIR/$filename"
    
    echo "📥 Downloading $filename..."
    if [ -f "$temp_file" ]; then
        echo "  ⚠️  File already exists in temp, skipping download"
    else
        wget -c "$url" -O "$temp_file" || {
            echo "❌ Download failed for $filename"
            return 1
        }
    fi
    
    echo "📦 Extracting $filename..."
    unzip -o "$temp_file" -d "$DATA_DIR" || {
        echo "❌ Extraction failed for $filename"
        return 1
    }
    
    echo "✅ $filename completed"
    echo ""
}

# Check existing files
echo "🔍 Checking existing files..."
if [ -d "$DATA_DIR/imageFiles" ]; then
    echo "  ✓ imageFiles already exists"
    read -p "  Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        SKIP_IMAGES=true
    fi
fi

if [ -d "$DATA_DIR/sequenceFiles" ]; then
    echo "  ✓ sequenceFiles already exists"
    read -p "  Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        SKIP_SEQUENCES=true
    fi
fi

echo ""

# Download files
if [ "$SKIP_IMAGES" != "true" ]; then
    download_and_extract "imageFiles.zip"
else
    echo "⏭️  Skipping imageFiles.zip (already exists)"
fi

if [ "$SKIP_SEQUENCES" != "true" ]; then
    download_and_extract "sequenceFiles.zip"
else
    echo "⏭️  Skipping sequenceFiles.zip (already exists)"
fi

# Download readme
echo "📥 Downloading readme and demo..."
wget -c "$BASE_URL/readme_and_demo.zip" -O "$TEMP_DIR/readme_and_demo.zip" || true
if [ -f "$TEMP_DIR/readme_and_demo.zip" ]; then
    unzip -o "$TEMP_DIR/readme_and_demo.zip" -d "$DATA_DIR"
    echo "✅ Readme downloaded"
fi

echo ""
echo "🎉 3DPW Download Complete!"
echo ""
echo "📊 Dataset Structure:"
ls -lh "$DATA_DIR"

echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "✅ Done!"
