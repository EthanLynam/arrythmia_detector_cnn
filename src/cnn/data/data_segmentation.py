import os
import random

# paths
SOURCE_DIR = "../../../created-images"
TEST_DIR = "test"
TRAINING_DIR = "training"
VALIDATION_DIR = "validation"

# ratios (70% training, 20% validation, 10% test)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# create output folders if they dont exist
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# iterate through folders in created-images
for class_folder in os.listdir(SOURCE_DIR):

    # avpids OTHER and EDITED as these are for viewing purposes,
    # we are not interested in training the CNN to detect OTHER,
    # EDITED is for easier viewing and comparison of augments.
    if class_folder in ["EDITED", "OTHER"]:
        continue

    class_path = os.path.join(SOURCE_DIR, class_folder) # creates path for current folder in created-images

    # all images in current folder
    images = os.listdir(class_path)
    random.shuffle(images)  # shuffle the images in memory to prevent bunvhes of same patients data

    # find splits (70 and 20 percent, remainder 10)
    total_images = len(images)
    train_end = int(TRAIN_RATIO * total_images)
    val_end = train_end + int(VAL_RATIO * total_images)

    # Split images into train, val, and test
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # used to associate the segment types with their folder directories
    folder_map = {
        TRAINING_DIR: train_images,
        VALIDATION_DIR: val_images,
        TEST_DIR: test_images
    }

    # loops through folder_map
    for folder_name, segment_images in folder_map.items():

        # loops through each file (image) in each segment (train, validate, test)
        for image in segment_images:

            # path to current image to be added, found in created-images
            source_image_path = os.path.join(class_path, image)

            # path to either (test, train, val)'s type folder (normal, paced etc.)
            destination_folder = os.path.join(folder_name, class_folder)
            os.makedirs(destination_folder, exist_ok=True) # creates folder if doesnt exist (for first run on machine)

            # joins destination folder to a file name (source images name found in created-images)
            destination_path = os.path.join(destination_folder, image)

            # links image inside created-image to new destination inside test train val,
            # which prevents copying the images and instead holds the same data in 2 places.
            try:
                os.link(source_image_path, destination_path)
            except FileExistsError:
                pass

print("Dataset successfully split into training, validation, and test sets using file links")
