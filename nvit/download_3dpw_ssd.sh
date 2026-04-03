#!/bin/bash

# 3DPW Dataset Downloader (SSD Storage + Symlink)
# Downloads the complete 3DPW dataset to SSD and links it to local data dir

set -e

echo "📥 3DPW Dataset Downloader (SSD Mode)"
echo "====================================="
echo ""

# Configuration
# Storage path on massive SSD
SSD_ROOT="/mnt/ssd_samsung_1/home/nkd/yangz_data/datasets"
SSD_3DPW="$SSD_ROOT/3DPW"

# Project data path (Symlink target)
LOCAL_DATA_DIR="/home/yangz/4D-Humans/data/3DPW"

TEMP_DIR="/mnt/ssd_samsung_1/home/nkd/yangz_data/temp_3dpw_download"
BASE_URL="https://virtualhumans.mpi-inf.mpg.de/3DPW"

# Create directories
mkdir -p "$SSD_3DPW"
mkdir -p "$TEMP_DIR"

echo "💾 Storage: $SSD_3DPW"
echo "🔗 Link:    $LOCAL_DATA_DIR"
echo "🌐 Source:  $BASE_URL"
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
    
    echo "📦 Extracting $filename to SSD..."
    unzip -o "$temp_file" -d "$SSD_3DPW" || {
        echo "❌ Extraction failed for $filename"
        return 1
    }
    
    echo "✅ $filename completed"
    echo ""
}

# Check existing files on SSD
echo "🔍 Checking existing files on SSD..."
if [ -d "$SSD_3DPW/imageFiles" ]; then
    echo "  ✓ imageFiles already exists on SSD"
    SKIP_IMAGES=true
fi

if [ -d "$SSD_3DPW/sequenceFiles" ]; then
    echo "  ✓ sequenceFiles already exists on SSD"
    SKIP_SEQUENCES=true
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
    unzip -o "$TEMP_DIR/readme_and_demo.zip" -d "$SSD_3DPW"
    echo "✅ Readme downloaded"
fi

# Create Symlinks
echo ""
echo "🔗 Setting up Symlinks..."

# Backup existing local directory if it's not a symlink and not empty
if [ -d "$LOCAL_DATA_DIR" ] && [ ! -L "$LOCAL_DATA_DIR" ]; then
    echo "⚠️  Existing local data directory found: $LOCAL_DATA_DIR"
    
    # Check if it has content we probably want to move/merge?
    # For safety, let's just mv it to backup
    BACKUP_DIR="${LOCAL_DATA_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "📦 Moving existing local data to backup: $BACKUP_DIR"
    mv "$LOCAL_DATA_DIR" "$BACKUP_DIR"
fi

# Create parent dir if not exists
mkdir -p "$(dirname "$LOCAL_DATA_DIR")"

# Create the symlink
if [ -L "$LOCAL_DATA_DIR" ]; then
    # It exists and is a symlink, verify where it points
    CURRENT_TARGET=$(readlink -f "$LOCAL_DATA_DIR")
    if [ "$CURRENT_TARGET" == "$SSD_3DPW" ]; then
        echo "✅ Symlink already correct: $LOCAL_DATA_DIR -> $SSD_3DPW"
    else
        echo "🔄 Updating symlink..."
        rm "$LOCAL_DATA_DIR"
        ln -s "$SSD_3DPW" "$LOCAL_DATA_DIR"
        echo "✅ Symlink updated: $LOCAL_DATA_DIR -> $SSD_3DPW"
    fi
else
    ln -s "$SSD_3DPW" "$LOCAL_DATA_DIR"
    echo "✅ Symlink created: $LOCAL_DATA_DIR -> $SSD_3DPW"
fi

echo ""
echo "🎉 3DPW Setup Complete on SSD!"
echo ""
echo "📊 Dataset Structure (SSD):"
ls -lh "$SSD_3DPW"

echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "✅ Done!"
