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
        print(f"Warning: No annotation found for image: {image_filename}.")
        if annotations:
            annotation_key = next(iter(annotations))
            print(f"Using annotation for image: {annotations[annotation_key]['filename']}")
        else:
            raise ValueError("No annotations provided in file.")

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
    import math
    n = len(regions_processed)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    # Flatten axes in case of multi-dimensional array
    axes = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            img, coords = regions_processed[i]
            ax.imshow(np.array(img))
            ax.set_title(f"Region {i+1}: {coords}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Process each region with OCR and build predictions
    from collections import Counter
    for idx, (region_image, coords) in enumerate(regions_processed):
        candidates = []
        confidences = []
        
        # Attempt 1: Original preprocessed image
        region_np = np.array(region_image)
        results1 = reader.readtext(region_np)
        if results1:
            best1 = max(results1, key=lambda x: x[2])
            cand1 = best1[1].strip()
            candidates.append(cand1)
            confidences.append(best1[2])
        
        # Attempt 2: Inverted image
        from PIL import ImageOps
        inverted = region_image.copy().convert("L")
        inverted = ImageOps.invert(inverted)
        inverted = inverted.convert("RGB")
        results2 = reader.readtext(np.array(inverted))
        if results2:
            best2 = max(results2, key=lambda x: x[2])
            cand2 = best2[1].strip()
            candidates.append(cand2)
            confidences.append(best2[2])
        
        # Attempt 3: Binarized (thresholded) image
        binarized = region_image.convert("L").point(lambda p: 255 if p > 128 else 0, mode="1")
        binarized = binarized.convert("RGB")
        results3 = reader.readtext(np.array(binarized))
        if results3:
            best3 = max(results3, key=lambda x: x[2])
            cand3 = best3[1].strip()
            candidates.append(cand3)
            confidences.append(best3[2])
        
        # Clean candidates (remove any fallback markers)
        cleaned_candidates = [cand.replace(" [inv]", "") for cand in candidates]
        # Use majority vote if available; otherwise choose the candidate with highest confidence.
        counter = Counter(cleaned_candidates)
        if counter:
            majority_candidate, count = counter.most_common(1)[0]
            if count >= 2:
                final_text = majority_candidate
            else:
                final_text = cleaned_candidates[confidences.index(max(confidences))]
        else:
            final_text = ""
        
        # Set final confidence as max confidence among attempts (if available)
        conf = max(confidences) if confidences else 0.0

        # Additional post-process: collapse repeated characters (if any)
        if final_text and len(set(final_text)) == 1 and len(final_text) > 1:
            final_text = final_text[0]

        # Save debug images (optional)
        region_image.save(f"debug_region_{idx+1}.png")
        inverted.save(f"debug_inverted_region_{idx+1}.png")
        binarized.save(f"debug_binarized_region_{idx+1}.png")

        predictions.append({
            'region_index': idx,
            'coordinates': {'x': coords[0], 'y': coords[1], 'width': coords[2], 'height': coords[3]},
            'prediction': final_text,
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

    # Combine predictions into a final result without hard coding characters
    # Mapping: die_no = region1 + region2, day = region3 + region4, shift = region5, year = region6, month = region7
    if len(predictions) >= 7:
        # Remove any fallback markers from the final result fields.
        final_result = {
            "die_no": (predictions[0]['prediction'] + predictions[1]['prediction']).replace(" [inv]", ""),
            "day": (predictions[2]['prediction'] + predictions[3]['prediction']).replace(" [inv]", ""),
            "shift": predictions[4]['prediction'].replace(" [inv]", ""),
            "year": predictions[5]['prediction'].replace(" [inv]", ""),
            "month": predictions[6]['prediction'].replace(" [inv]", "")
        }
        combined_string = (
            final_result["die_no"]
            + "#" + final_result["day"]
            + "#" + final_result["shift"]
            + "#" + final_result["year"]
            + "#" + final_result["month"]
        )
        print("\nFinal Result:")
        print(combined_string)
        print("\nJSON Output:")
        json_output = json.dumps(final_result, indent=4)
        print(json_output)
        
        # Save the JSON output to a file
        with open("output.json", "w") as f:
            f.write(json_output)
    else:
        print("Not enough predictions to form final result.")

    return predictions


if __name__ == "__main__":
    image_path = r"D:\hik\hik3\Image_w4024_h3036_fn692.png"
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