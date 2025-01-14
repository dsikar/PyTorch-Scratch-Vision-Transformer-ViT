#!/bin/bash

# Set the directory to check (user's home directory)
DIR="$HOME"

# Find and display the largest files
echo "Finding largest files in $DIR..."
find "$DIR" -type f -exec du -h {} + 2>/dev/null | sort -rh | head -n 20

# Find and display the largest directories
echo -e "\nFinding largest directories in $DIR..."
du -ah "$DIR" 2>/dev/null | sort -rh | head -n 20
