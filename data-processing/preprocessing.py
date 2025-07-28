'''
PROCESS:
NII -> PNG (ROT 90) -> 256*256 -> SELECTED SLICES
'''

import os
import numpy as np
import nibabel
import shutil
from PIL import Image
import random


def convert_nii_to_png(input_path, output_path, rotation_angle=90):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    source_files = os.listdir(input_path)
    slice_counter = 0

    modalities = ['t1', 't1ce', 't2', 'flair']

    for file_name in source_files:
        if file_name.endswith('.nii.gz'):
            modality = file_name.split('_')[0].lower()
            name2 = file_name.split('_')[1].lower()
            idx = name2.split('.')[0].lower()

            if modality not in modalities:
                continue

            image_array = nibabel.load(os.path.join(input_path, file_name)).get_fdata()

            if len(image_array.shape) in [3, 4]:
                total_slices = image_array.shape[2]

                for current_slice in range(total_slices):
                    data = rotate_image(image_array[:, :, current_slice], rotation_angle)
                    save_image(output_path, modality, idx, current_slice, data)
                    slice_counter += 1

def rotate_image(image, rotation_angle):
    if rotation_angle % 360 == 0:
        return image
    return np.rot90(image, rotation_angle // 90)

def save_image(output_path, modality, idx, current_slice, data):
    print('Saving image...')
    normalized_data = (data - data.min()) / np.maximum((data.max() - data.min()), 1e-5)
    uint8_data = (normalized_data * 255).astype(np.uint8)
    image_name = f'{modality}_{idx}_{current_slice}.png'
    
    image = Image.fromarray(uint8_data)
    image.save(os.path.join(output_path, image_name))
    
    print('Saved.')


def resize_images(input_dir, output_dir, target_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)

    print('Resizing images...')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path_image = os.path.join(input_dir, filename)
            
            img = Image.open(input_path_image).convert('L')

            resized_img = img.resize(target_size)

            output_path_image = os.path.join(output_dir, filename)
            
            resized_img.save(output_path_image)
            
            print(f'Resized: {filename}')



def select_slices(input_selected, output_selected):
    os.makedirs(output_selected, exist_ok=True)

    input_slices = os.listdir(input_selected)

    for slice in input_slices:
        if slice.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path_slices = os.path.join(input_selected, slice)

            image = Image.open(input_path_slices).convert('L')
    
            pixels = list(image.getdata())

            non_black_pixel_count = pixels.count(0)

            width, height = image.size

            total_area_pixels = width * height

            area_excluding_black_pixels = total_area_pixels - non_black_pixel_count

            # AREA UP 30% -> 19666
            if area_excluding_black_pixels >= 19666:
                selected_slices = os.path.join(output_selected, slice)
                shutil.copy2(input_path_slices, selected_slices)
                print(f"Selected slices finished for image {slice}")


def organize_files(source_directory, destination_directory, train_ratio=0.75, test_ratio=0.15):
    os.makedirs(destination_directory, exist_ok=True)

    for split in ["train", "test", "valid"]:
        os.makedirs(os.path.join(destination_directory, split, "T1_gt"), exist_ok=True)
        os.makedirs(os.path.join(destination_directory, split, "T1CE_gt"), exist_ok=True)
        os.makedirs(os.path.join(destination_directory, split, "T2_gt"), exist_ok=True)
        os.makedirs(os.path.join(destination_directory, split, "FL_gt"), exist_ok=True)

    file_list = os.listdir(source_directory)

    t1_files = [file for file in file_list if "t1_" in file]
    t1ce_files = [file for file in file_list if "t1ce_" in file]
    t2_files = [file for file in file_list if "t2_" in file]
    flair_files = [file for file in file_list if "flair_" in file]

    max_slices = min(len(t1_files), len(t1ce_files), len(t2_files), len(flair_files))

    train_count = int(max_slices * train_ratio)
    test_count = int(max_slices * test_ratio)
    valid_count = max_slices - train_count - test_count
    
    def copy_files(files, train_count, test_count, valid_count, modality):
        random.shuffle(files)
        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        valid_files = files[train_count + test_count:train_count + test_count + valid_count]

        for file in train_files:
            source_path = os.path.join(source_directory, file)
            destination_path = os.path.join(destination_directory, "train", f"{modality}_gt", file)
            shutil.copy(source_path, destination_path)
            print(f"Copied {file} to {destination_path}")

        for file in test_files:
            source_path = os.path.join(source_directory, file)
            destination_path = os.path.join(destination_directory, "test", f"{modality}_gt", file)
            shutil.copy(source_path, destination_path)
            print(f"Copied {file} to {destination_path}")

        for file in valid_files:
            source_path = os.path.join(source_directory, file)
            destination_path = os.path.join(destination_directory, "valid", f"{modality}_gt", file)
            shutil.copy(source_path, destination_path)
            print(f"Copied {file} to {destination_path}")

    copy_files(t1_files, train_count, test_count, valid_count, "T1")
    copy_files(t1ce_files, train_count, test_count, valid_count, "T1CE")
    copy_files(t2_files, train_count, test_count, valid_count, "T2")
    copy_files(flair_files, train_count, test_count, valid_count, "FL")
    
    print("Organization completed.")


data_directory = 'path/to/data_directory'

def main():
    
    input_path = os.path.join(data_directory, 'path/to/data_nii')
    output_path = os.path.join(data_directory, 'path/to/data_png')
    rotation_angle = 90

    convert_nii_to_png(input_path, output_path, rotation_angle)

    input_dir = os.path.join(data_directory, 'path/to/data_png')
    output_dir = os.path.join(data_directory, 'path/to/data_png_T256')
    resize_images(input_dir, output_dir)
    
    input_selected = os.path.join(data_directory, 'path/to/data_png_T256')
    output_selected = os.path.join(data_directory, 'path/to/data_png_T256_selected')
    select_slices(input_selected, output_selected)
    
    source_directory = os.path.join(data_directory, 'path/to/data_png_T256_selected')
    destination_directory = os.path.join(data_directory, 'path/to/data_png_T256_selected_AREA_30')
    organize_files(source_directory, destination_directory, train_ratio=0.75, test_ratio=0.15, valid_ratio=0.10)

    print('Finished converting, selecting, and organizing images')


if __name__ == "__main__":
    main()


