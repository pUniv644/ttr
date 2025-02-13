# run.py
import os
from char_recognition import CharacterRecognizer
from config import *

def main():
    # Initialize recognizer with annotations
    annotations_path = os.path.join(DATA_DIR, 'annotations', 'annotations.json')
    image_path = os.path.join(DATA_DIR, 'images', 'Image_w4024_h3036_fn1.png')
    
    recognizer = CharacterRecognizer(annotations_path)
    
    # 1. Extract individual character images
    char_images_dir = os.path.join(OUTPUT_DIR, 'char_images')
    print(f"Extracting character images to: {char_images_dir}")
    recognizer.extract_and_save_chars(image_path, char_images_dir)
    
    # 2. Visualize bounding boxes
    vis_output = os.path.join(OUTPUT_DIR, 'visualization', 'annotated_image.png')
    print("Creating visualization...")
    recognizer.visualize_boxes(image_path, vis_output)
    
    # 3. Print region information
    print("\nRegion Information:")
    regions = recognizer.get_labeled_regions()
    for region in regions:
        print(f"Region {region['region_id']}: "
              f"Character '{region['label']}' at position {region['bbox']}")

if __name__ == "__main__":
    main()