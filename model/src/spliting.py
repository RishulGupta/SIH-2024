import os
import shutil
import random


import os

bw_folder = os.path.join('..', 'data', 'colorization', 'gray')
color_folder = os.path.join('..', 'data', 'colorization', 'color')

print("Checking paths...")
print("bw_folder:", os.path.abspath(bw_folder), "Exists:", os.path.exists(bw_folder))
print("color_folder:", os.path.abspath(color_folder), "Exists:", os.path.exists(color_folder))


# Define paths (UPDATE THESE)
bw_folder = os.path.join('..', 'data', 'colorization', 'gray')
color_folder = os.path.join('..', 'data', 'colorization', 'color')
output_dir = os.path.join('..', 'data', 'colorization', 'output')

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1!"

# Function to split images
def split_and_copy_images(src_folder, dest_folder):
    if not os.path.exists(src_folder):
        print(f"❌ Error: Source folder '{src_folder}' does not exist!")
        return

    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f"⚠️ Warning: No images found in '{src_folder}'!")
        return
    
    random.shuffle(images)

    total_images = len(images)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * val_ratio)

    sets = {
        "train": images[:train_split],
        "val": images[train_split:train_split + val_split],
        "test": images[train_split + val_split:]
    }

    for set_name, file_list in sets.items():
        set_path = os.path.join(dest_folder, set_name)
        os.makedirs(set_path, exist_ok=True)

        print(f"📂 Moving {len(file_list)} images to {set_path}...")

        for file in file_list:
            src_path = os.path.join(src_folder, file)
            dest_path = os.path.join(set_path, file)
            try:
                shutil.copy(src_path, dest_path)
            except Exception as e:
                print(f"❌ Failed to copy {file}: {e}")

# Create output directories
os.makedirs(output_dir, exist_ok=True)

# Process both folders
print("🚀 Splitting images...")
split_and_copy_images(bw_folder, os.path.join(output_dir, "bw"))
split_and_copy_images(color_folder, os.path.join(output_dir, "color"))

print("✅ Splitting completed successfully!")
