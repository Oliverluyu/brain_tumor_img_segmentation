import os
import shutil
from os.path import join
import numpy as np

# Paths setup
root_dir = 'datasets/classification'
train_dir = join(root_dir, 'train')
val_dir = join(root_dir, 'validation')

# Create the validation directory if it doesn't exist
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Percentage of images to move to validation set
val_split = 0.05

for tumor_type in os.listdir(train_dir):
    # Check if the folder exists in the validation directory; if not, create it
    tumor_val_dir = join(val_dir, tumor_type)
    if not os.path.exists(tumor_val_dir):
        os.makedirs(tumor_val_dir)

    # List all images in the current tumor type directory
    tumor_images = os.listdir(join(train_dir, tumor_type))
    total_images = len(tumor_images)

    # Calculate the number of images to move
    num_val_images = int(np.floor(val_split * total_images))

    # Randomly select images to move
    val_images = np.random.choice(tumor_images, size=num_val_images, replace=False)

    # Move selected images to the corresponding validation directory
    for image in val_images:
        src_path = join(train_dir, tumor_type, image)
        dst_path = join(tumor_val_dir, image)
        shutil.move(src_path, dst_path)

    print(f'Moved {num_val_images} images from {tumor_type} to validation set.')
