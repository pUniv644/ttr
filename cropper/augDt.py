import os
import shutil
import numpy as np
from PIL import Image
import albumentations as A
import cv2
from tqdm import tqdm

def create_directory_structure(base_dir):
    """Create the required directory structure."""
    # Create main splits
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        
        # Create class directories (0-9 and A-Z)
        for i in range(10):  # 0-9
            os.makedirs(os.path.join(split_dir, str(i)), exist_ok=True)
        
        for i in range(65, 91):  # A-Z
            os.makedirs(os.path.join(split_dir, chr(i)), exist_ok=True)

def load_and_preprocess_image(image_path):
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image, save_path):
    """Save the augmented image."""
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)

def create_augmentation_pipeline():
    """Create an augmentation pipeline."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-15, 15),
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),
    ])

def generate_dataset(source_dir, base_dir, num_augmentations=5):
    """Generate the dataset with augmentations."""
    # Create directory structure
    create_directory_structure(base_dir)
    
    # Create augmentation pipeline
    aug_pipeline = create_augmentation_pipeline()
    
    # Map region numbers/letters to actual classes
    region_mapping = {
        '0': 'S',
        '1': '1',
        '2': '1',
        '3': '3',
        '4': 'A',
        '5': '5',
        '6': 'A'
    }
    
    # Process each source image
    for region_num in range(7):
        image_path = os.path.join(source_dir, f'region_{region_num}.png')
        print(f"Processing region {region_num}: {image_path}")
        
        try:
            # Verify file exists
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
                
            # Get the actual class label
            class_label = region_mapping[str(region_num)]
            
            # Load the source image
            image = load_and_preprocess_image(image_path)
            
            # Generate augmented images for both train and val splits
            for split in ['train', 'val']:
                split_dir = os.path.join(base_dir, split, class_label)
                os.makedirs(split_dir, exist_ok=True)
                
                # Save the original image
                save_path = os.path.join(split_dir, f'original_{region_num}.png')
                save_image(image, save_path)
                
                # Generate augmented versions
                for aug_idx in tqdm(range(num_augmentations), 
                                  desc=f"Generating augmentations for region {region_num} {split}"):
                    augmented = aug_pipeline(image=image)['image']
                    save_path = os.path.join(split_dir, f'aug_{region_num}_{aug_idx}.png')
                    save_image(augmented, save_path)
                    
        except Exception as e:
            print(f"Error processing region {region_num}: {str(e)}")
            continue

def main():
    # Source directory containing the region_*.png files
    source_dir = r"D:\gpu-ocr\cropped"
    
    # Base directory for the dataset (will create in the same parent directory)
    base_dir = os.path.join(os.path.dirname(source_dir), 'data_dir')
    
    # Generate the dataset
    print(f"Starting dataset generation...")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {base_dir}")
    
    generate_dataset(source_dir, base_dir)
    
    print("Dataset generation completed!")

if __name__ == "__main__":
    main()