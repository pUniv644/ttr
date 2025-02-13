import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import json
import os
import random
from model import SimpleCharClassifier

def preprocess_region(image):
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Convert back to grayscale
    image = image.convert('L')
    
    return image

def load_model(model_path):
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCharClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_regions(model, image_path, annotations_file):
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Load and check image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    original_image = Image.open(image_path)
    
    # Prepare transform for cropped regions
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Get device
    device = next(model.parameters()).device
    
    # Get filename without path
    image_filename = os.path.basename(image_path)
    
    # Find matching annotation
    annotation_key = None
    for key in annotations:
        if annotations[key]['filename'] == image_filename:
            annotation_key = key
            break
    
    if annotation_key is None:
        raise ValueError(f"No annotation found for image: {image_filename}")
        
    # Process each region
    regions = annotations[annotation_key]['regions']
    predictions = []
    
    for idx, region in enumerate(regions):
        # Extract region coordinates
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        width = region['shape_attributes']['width']
        height = region['shape_attributes']['height']
        
        # Add padding to region
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = width + 2 * padding
        height = height + 2 * padding
        
        # Crop region from original image
        region_image = original_image.crop((x, y, x + width, y + height))
        
        # Preprocess region
        region_image = preprocess_region(region_image)
        
        # Transform
        region_tensor = transform(region_image).unsqueeze(0).to(device)
        
        # Predict with ensemble
        with torch.no_grad():
            # Make multiple predictions with small random adjustments
            pred_labels = []
            pred_confidences = []
            
            for _ in range(5):
                # Add small random rotation
                angle = random.uniform(-5, 5)
                rotated = region_tensor.clone()
                rotated = TF.rotate(rotated, angle)
                
                output = model(rotated)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                pred_labels.append(predicted.item())
                pred_confidences.append(confidence.item())
            
            # Use majority voting
            final_pred = max(set(pred_labels), key=pred_labels.count)
            final_conf = sum(pred_confidences) / len(pred_confidences)
        
        # Convert prediction to class label
        class_labels = ['S', '1', '3', '5', 'A']
        predictions.append({
            'region_index': idx,
            'coordinates': {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            },
            'prediction': class_labels[final_pred],
            'confidence': final_conf
        })
    
    return predictions

if __name__ == "__main__":
    # Example usage
    model_path = "models/simple_char_classifier.pth"
    image_path = "data/images/Image_w4024_h3036_fn1.png"
    annotations_file = "data/annotations/annotations.json"
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(model_path)
        
        # Process image regions and make predictions
        print(f"Processing regions in {image_path}...")
        predictions = predict_regions(model, image_path, annotations_file)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        for pred in predictions:
            print(f"Region {pred['region_index'] + 1}:")
            print(f"  Position: (x={pred['coordinates']['x']}, y={pred['coordinates']['y']})")
            print(f"  Size: {pred['coordinates']['width']}x{pred['coordinates']['height']}")
            print(f"  Predicted: {pred['prediction']}")
            print(f"  Confidence: {pred['confidence']:.2%}")
            print()
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")