import shutil
import os
from pathlib import Path
import subprocess

def export_project():
    source_dir = Path("/home/node/.openclaw/workspace/BhashaLLM")
    dest_dir = Path(os.path.expanduser("~/Desktop/BhashaLLM_Export"))
    
    print(f"Exporting project from {source_dir} to {dest_dir}...")
    
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    
    # Directories to copy
    dirs_to_copy = [
        "data", # Copy all data (including processed)
        "models/ocr_adapters", # Only adapters
        "report",
        "models/qwen_adapters" # If exists
    ]
    
    # Files to copy
    extensions = ["*.py", "*.md", "*.txt", "*.jsonl", "*.yml"]
    
    # Copy directories
    for d in dirs_to_copy:
        src = source_dir / d
        dst = dest_dir / d
        if src.exists():
            print(f"Copying {d}...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            
    # Copy files
    for ext in extensions:
        for f in source_dir.glob(ext):
            print(f"Copying {f.name}...")
            shutil.copy2(f, dest_dir)
            
    # Special copy for generated artifacts in brain if needed?
    # No, they are likely in source_dir or not needed.
    
    print("Export complete.")
    
if __name__ == "__main__":
    export_project()
