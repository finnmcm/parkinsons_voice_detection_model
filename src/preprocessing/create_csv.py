import pandas as pd 
import os


df = pd.DataFrame(columns=['filepath', 'label'])
def add_folder(folder_path: str, start: int):
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
    print(folder_path + ":" + str(len(files)) + " files")
    for count, filename in enumerate(files, start=start):
        filepath = folder_path + "/" + filename
        args = filename.split('_')
        hyScore = args[1]
        if hyScore == 'hc':
            hyScore = hyScore.upper()
        else:
            hyScore = "HY" + str(hyScore)
            
        df.loc[count] = [filepath, hyScore]
        
add_folder('data/raw/HC', len(df))
add_folder('data/raw/HY2', len(df))
add_folder('data/raw/HY3', len(df))
add_folder('data/raw/HY4', len(df))
df.drop(index=0, inplace=True)
df.to_csv('metadata.csv', index=False)