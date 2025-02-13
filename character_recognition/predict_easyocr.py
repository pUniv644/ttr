import json
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr

def preprocess_region(image):
    # Convert to grayscale then back to RGB
    gray = image.convert("L")
    # Increase contrast more aggressively
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    # Optionally apply a slight smoothing if needed
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    # Convert back to RGB so that easyOCR receives expected format
    result = sharpened.convert("RGB")
    return result

def predict_regions(image_path, annotations_file):
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Check image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    original_image = Image.open(image_path)
    image_filename = os.path.basename(image_path)

    # Find corresponding annotation
    annotation_key = None
    for key in annotations:
        if annotations[key]['filename'] == image_filename:
            annotation_key = key
            break
    if annotation_key is None:
        raise ValueError(f"No annotation found for image: {image_filename}")

    regions = annotations[annotation_key]['regions']
    predictions = []

    # Initialize easyocr reader
    reader = easyocr.Reader(['en'])
    
    # (Assumed ground truth mapping; update with your actual labels.)
    ground_truth = {
        0: "S",
        1: "1",
        2: "1",
        3: "3",
        4: "A",
        5: "5",
        6: "A"
    }

    # Process regions: cropping and preprocessing
    regions_processed = []  # List of tuples: (processed PIL image, (x, y, width, height))
    for region in regions:
        # Extract region coordinates with padding
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        width = region['shape_attributes']['width']
        height = region['shape_attributes']['height']
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        width += 2 * padding
        height += 2 * padding

        # Crop and preprocess region; no conversion to grayscale
        region_image = original_image.crop((x, y, x + width, y + height))
        region_image = preprocess_region(region_image)
        # Remove the line below if your image is already in RGB:
        # region_image = region_image.convert('RGB')
        regions_processed.append((region_image, (x, y, width, height)))

    # Display preprocessed regions using matplotlib before OCR prediction
    import matplotlib.pyplot as plt
    n = len(regions_processed)
    fig, axes = plt.subplots(n, 1, figsize=(5, 3 * n))
    if n == 1:  # Ensure axes is iterable
        axes = [axes]
    for i, ax in enumerate(axes):
        img, coords = regions_processed[i]
        ax.imshow(np.array(img))
        ax.set_title(f"Region {i+1}: Position: {coords}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Process each region with OCR and build predictions
    for idx, (region_image, coords) in enumerate(regions_processed):
        region_np = np.array(region_image)
        # Run easyocr on the region
        results = reader.readtext(region_np)
        if results:
            # Select the prediction with the highest confidence
            best = max(results, key=lambda x: x[2])
            text = best[1].strip()
            conf = best[2]
        else:
            text = ""
            conf = 0.0

        # Inside your loop after processing each region (for debugging):
        region_image.save(f"debug_region_{idx+1}.png")

        predictions.append({
            'region_index': idx,
            'coordinates': {'x': coords[0], 'y': coords[1], 'width': coords[2], 'height': coords[3]},
            'prediction': text,
            'confidence': conf
        })

    # Print detailed results table
    print("\nDetailed Results:")
    print("Region | Predicted | Actual | Confidence")
    print("-------|-----------|--------|------------")
    for pred in predictions:
        idx = pred['region_index']
        predicted_text = pred['prediction']
        conf_percent = f"{pred['confidence']*100:.2f}%"
        actual = ground_truth.get(idx, "N/A")
        mark = "✓" if predicted_text == actual else "✗"
        print(f"{idx+1:<6} | {predicted_text:<9} | {actual:<6} | {conf_percent:<10} {mark}")

    return predictions

if __name__ == "__main__":
    image_path = "data/images/Image_w4024_h3036_fn1.png"
    annotations_file = "data/annotations/annotations.json"
    
    try:
        print("Processing regions using easyOCR...")
        predictions = predict_regions(image_path, annotations_file)
        print("\nPrediction Results (easyOCR):")
        print("-" * 50)
        for pred in predictions:
            print(f"Region {pred['region_index'] + 1}:")
            print(f"  Position: (x={pred['coordinates']['x']}, y={pred['coordinates']['y']})")
            print(f"  Size: {pred['coordinates']['width']}x{pred['coordinates']['height']}")
            print(f"  Predicted Text: {pred['prediction']}")
            print()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")