import os
import random
import shutil
from os.path import join
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split


def classfication_split():

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


def segmentation_split():

    # Define the source and target directories
    source_images = 'E:/Oliver/computer science homework/CS5260 DL2/project/brain_tumour/images'
    source_masks = 'E:/Oliver/computer science homework/CS5260 DL2/project/brain_tumour/masks'
    target_base = 'datasets/segmentation'

    # Get all filenames from the source directories
    images = os.listdir(source_images)
    masks = os.listdir(source_masks)
    print('get images and masks from source directory successfully!')

    # Ensure the filenames for images and masks match and are sorted
    images.sort()
    masks.sort()
    assert images == masks, "Image and mask filenames do not match."

    # Split the dataset
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        images, masks, test_size=0.10, random_state=42)

    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=(15 / 85),
        random_state=42)  # 15% of 85% is approximately 15% of the whole

    print('train, validation and test split successful!')

    # Function to copy files to their new location
    def copy_files(files, source_dir, target_dir):
        for file in files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))

    # Create directories and subdirectories for the new structure
    for split in ['train', 'validation', 'test']:
        for folder in ['images', 'masks']:
            for present in ['tumour-present', 'no-tumour']:
                os.makedirs(os.path.join(target_base, split, folder, present), exist_ok=True)

    # Copy files to their new respective directories
    copy_files(train_images, source_images, os.path.join(target_base, 'train', 'images', 'tumour-present'))
    copy_files(train_masks, source_masks, os.path.join(target_base, 'train', 'masks', 'tumour-present'))
    copy_files(val_images, source_images, os.path.join(target_base, 'validation', 'images', 'tumour-present'))
    copy_files(val_masks, source_masks, os.path.join(target_base, 'validation', 'masks', 'tumour-present'))
    copy_files(test_images, source_images, os.path.join(target_base, 'test', 'images', 'tumour-present'))
    copy_files(test_masks, source_masks, os.path.join(target_base, 'test', 'masks', 'tumour-present'))

    print("Files have been successfully split and copied to the new directory structure.")


def segmentation_notumour_choice(target_file_num):

    # Set the source and target directories
    source_dir = 'datasets/classification/train/notumor'
    target_dir = 'datasets/segmentation/train/images/no-tumour'

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # List all files in the source directory
    all_files = os.listdir(source_dir)

    # Randomly select 920 files
    selected_files = random.sample(all_files, target_file_num)

    # Copy the selected files to the target directory
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(source_path, target_path)

    print(
        f"{len(selected_files)} no-tumour images have been successfully copied to the segmentation folder.")


def segmentation_generate_notumour_mask(split):
    # Define the source and target directories
    root_dir = 'datasets/segmentation'
    source_images_dir = join(root_dir, split, 'images/no-tumour')
    target_masks_dir = join(root_dir, split, 'masks/no-tumour')

    # Ensure the target directory exists
    os.makedirs(target_masks_dir, exist_ok=True)

    # List all files in the source images directory
    image_files = os.listdir(source_images_dir)

    # Generate a blank mask for each image
    for image_file in image_files:
        # Open the image to find its dimensions
        image_path = os.path.join(source_images_dir, image_file)
        with Image.open(image_path) as img:
            # Create a blank (all-zero) image with the same dimensions
            mask = Image.new('L', img.size, 0)  # 'L' mode for (8-bit pixels, black and white)

        # Save the blank mask with the same filename in the target directory
        mask.save(os.path.join(target_masks_dir, image_file))

    print(f"Generated blank masks for {len(image_files)} no-tumour images.")


# classification preparation
# classfication_split()

# segmentation preparation
# segmentation_split()

# random select no-tumour images from classification to segmentation dataset
# segmentation_notumour_choice(target_file_num=920)

# generate masks for no-tumour images in segmentation task
# segmentation_generate_notumour_mask('train')
# segmentation_generate_notumour_mask('validation')
# segmentation_generate_notumour_mask('test')