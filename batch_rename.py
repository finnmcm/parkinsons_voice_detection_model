#!/usr/bin/env python3
import os
import argparse

def batch_rename(folder_path: str, pattern: str, start_index: int, dry_run: bool):
    """
    Rename all files in folder_path according to pattern.
    
    pattern is a Python format string that can use:
      {index}    – a running number (starting at start_index)
      {name}     – original filename without extension
      {ext}      – original file extension (including the leading dot)
    
    e.g. pattern = "photo_{index:03d}{ext}"   → photo_001.jpg, photo_002.png, …
         pattern = "{name}_backup{ext}"       → image_backup.jpg, doc_backup.pdf, …
    """
    # Get a sorted list so renaming order is predictable
    files = sorted(f for f in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, f)))
    
    for count, filename in enumerate(files, start=start_index):
        name, ext = os.path.splitext(filename)
        args = name.split("_")
        patientName = args[0]
        hyScore = args[2]
        new_name = pattern.format(patientName=patientName, index=count, hy=hyScore, ext=ext)
        
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        
        print(f"{'DRY RUN:' if dry_run else ''} {filename}  →  {new_name}")
        if not dry_run:
            os.rename(src, dst)

if __name__ == "__main__":
  batch_rename("data/raw/dataset/SpontaneousDialogue/PD", "{patientName}_{hy}_dialogue.wav", 0, False)