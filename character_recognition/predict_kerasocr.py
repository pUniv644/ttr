import json
import os
from PIL import Image, ImageEnhance
import numpy as np
import keras_ocr
import tensorflow as tf
from tensorflow.python.keras.layers import Dense as KerasDense

# Monkey patch the Dense layer to ignore the "weights" keyword
class PatchedDense(KerasDense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('weights', None)
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        # If inputs is a tuple, select its first element
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        return super().call(inputs, *args, **kwargs)

tf.keras.layers.Dense = PatchedDense

def preprocess_region(image):
    # Convert to grayscale if needed and enhance contrast
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

    # Check if image exists
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

    # Initialize keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    for idx, region in enumerate(regions):
        # Extract region coordinates with added padding
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
        # Convert the grayscale image back to RGB
        region_image = region_image.convert('RGB')
        region_np = np.array(region_image)

        # Run keras-ocr on the region (pipeline expects a list of images)
        prediction_groups = pipeline.recognize([region_np])
        if prediction_groups and len(prediction_groups[0]) > 0:
            text = " ".join([item[0] for item in prediction_groups[0]])
        else:
            text = ""

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
        print("Processing regions using keras-ocr...")
        predictions = predict_regions(image_path, annotations_file)
        print("\nPrediction Results (keras-ocr):")
        print("-" * 50)
        for pred in predictions:
            print(f"Region {pred['region_index'] + 1}:")
            print(f"  Position: (x={pred['coordinates']['x']}, y={pred['coordinates']['y']})")
            print(f"  Size: {pred['coordinates']['width']}x{pred['coordinates']['height']}")
            print(f"  Predicted Text: {pred['prediction']}")
            print()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")