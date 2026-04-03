#!/bin/bash
# Safe Cleanup Utility for NViT Project
# Uses 'gio trash' instead of 'rm' to prevent permanent data loss.

TARGET=$1

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <file_or_directory>"
    exit 1
fi

if command -v gio &> /dev/null; then
    echo "♻️ Moving $TARGET to system trash..."
    /opt/anaconda3/bin/gio trash "$TARGET"
else
    echo "⚠️ 'gio' not found. Please delete manually or install trash-cli."
    exit 1
fi
