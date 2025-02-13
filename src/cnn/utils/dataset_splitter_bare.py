import os
import random
from PIL import Image

# paths
SOURCE_DIR = "../../../created_images"
BASE_DIR = "../data_bare"
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAINING_DIR = os.path.join(BASE_DIR, "training")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")

# ratios (70% training, 20% validation, 10% test)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# create output folders if they dont exist
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def crop_top(image_path, output_path, crop_percent=0.14):
    """Crops the top portion of an image and saves the new image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            crop_height = int(height * crop_percent)
            cropped_img = img.crop((0, crop_height, width, height))  # Crop top
            cropped_img.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# iterate through folders in created-images
for class_folder in os.listdir(SOURCE_DIR):
    
    if class_folder in ["EDITED", "OTHER"]:
        continue

    class_path = os.path.join(SOURCE_DIR, class_folder)
    images = os.listdir(class_path)
    random.shuffle(images)  # Shuffle the images

    # Downsample to a max of 3000 images if necessary
    if len(images) > 3000:
        images = random.sample(images, 3000)

    total_images = len(images)
    train_end = int(TRAIN_RATIO * total_images)
    val_end = train_end + int(VAL_RATIO * total_images)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    folder_map = {
        TRAINING_DIR: train_images,
        VALIDATION_DIR: val_images,
        TEST_DIR: test_images
    }

    for folder_name, segment_images in folder_map.items():
        destination_folder = os.path.join(folder_name, class_folder)
        os.makedirs(destination_folder, exist_ok=True)

        for image in segment_images:
            source_image_path = os.path.join(class_path, image)
            destination_path = os.path.join(destination_folder, image)

            try:
                crop_top(source_image_path, destination_path)
            except Exception as e:
                print(f"Skipping {image} due to error: {e}")

print("Dataset successfully split into training, validation, and test sets with cropped images.")
