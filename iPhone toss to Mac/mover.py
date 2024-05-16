# mover.py
from pathlib import Path

# Source directory to monitor
src_dir = Path("/Path/")

# Destination directory to move the files
dst_dir = Path("/Path/")

# Create the destination directory if it doesn't exist
dst_dir.mkdir(parents=True, exist_ok=True)

# Monitor the source directory for new .txt files
while True:
    for file_path in src_dir.glob("*.txt"):
        # Move the file to the destination directory
        dst_file_path = dst_dir / file_path.name
        shutil.move(str(file_path), str(dst_file_path))
        print(f"Moved {file_path.name} to {dst_file_path}")
