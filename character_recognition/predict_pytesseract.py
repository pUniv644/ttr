import json
import os
from PIL import Image, ImageEnhance
import pytesseract

def preprocess_region(image):
    # Convert to grayscale if not already and enhance contrast
    if image.mode != 'L':
        image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    image = image.convert('L')
    return image

def predict_regions(image_path, annotations_file):
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Check image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    original_image = Image.open(image_path)
    image_filename = os.path.basename(image_path)

    # Find matching annotation
    annotation_key = None
    for key in annotations:
        if annotations[key]['filename'] == image_filename:
            annotation_key = key
            break
    if annotation_key is None:
        raise ValueError(f"No annotation found for image: {image_filename}")

    regions = annotations[annotation_key]['regions']
    predictions = []
    for idx, region in enumerate(regions):
        # Extract region coordinates and add padding
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        width = region['shape_attributes']['width']
        height = region['shape_attributes']['height']
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        width += 2 * padding
        height += 2 * padding

        # Crop and preprocess region
        region_image = original_image.crop((x, y, x + width, y + height))
        region_image = preprocess_region(region_image)

        # Use pytesseract to perform OCR
        text = pytesseract.image_to_string(region_image)

        predictions.append({
            'region_index': idx,
            'coordinates': {'x': x, 'y': y, 'width': width, 'height': height},
            'prediction': text.strip()
        })

    return predictions

if __name__ == "__main__":
    image_path = "data/images/Image_w4024_h3036_fn1.png"
    annotations_file = "data/annotations/annotations.json"
    
    try:
        print("Processing regions using pytesseract...")
        predictions = predict_regions(image_path, annotations_file)
        print("\nPrediction Results (pytesseract):")
        print("-" * 50)
        for pred in predictions:
            print(f"Region {pred['region_index'] + 1}:")
            print(f"  Position: (x={pred['coordinates']['x']}, y={pred['coordinates']['y']})")
            print(f"  Size: {pred['coordinates']['width']}x{pred['coordinates']['height']}")
            print(f"  Predicted Text: {pred['prediction']}")
            print()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")