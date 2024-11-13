import os
import shutil

# Define paths
dataset_path = "/Users/mac/Documents/pjlab/repo/flickr8k/Flicker8k_Dataset"
caption_file = "resource/flicker8k/caption.txt"
validation_folder = "resource/flicker8k/validation"

# Create validation folder if it doesn't exist
os.makedirs(validation_folder, exist_ok=True)

# Read the caption file and extract filenames
with open(caption_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Extract filenames from the caption file
filenames = []
for line in lines:
    if line.strip() and not line.startswith('|'):
        parts = line.split('\t')
        if len(parts) > 0:
            filename = parts[0].strip()
            filenames.append(filename)
            print(filename)
            print(parts[1])

# Copy files to the validation folder
for filename in filenames:
    src = os.path.join(dataset_path, filename + ".jpg")
    dst = os.path.join(validation_folder, filename + ".jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {filename} to validation folder.")
    else:
        print(f"File {filename} not found in dataset path.")