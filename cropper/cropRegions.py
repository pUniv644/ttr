import json
import os
from PIL import Image

def process_json(json_data):
    output_folder = 'cropped'
    os.makedirs(output_folder, exist_ok=True)

    for image_key in json_data:
        image_info = json_data[image_key]
        filename = r"D:\hik\hk2\Image_w4024_h3036_fn1.png"
        regions = image_info['regions']

        try:
            original_image = Image.open(filename)
        except FileNotFoundError:
            print(f"Error: {filename} not found. Skipping.")
            continue

        for index, region in enumerate(regions):
            shape = region['shape_attributes']
            x, y = shape['x'], shape['y']
            width, height = shape['width'], shape['height']
            box = (x, y, x + width, y + height)
            cropped_image = original_image.crop(box)

            # basename, ext = os.path.splitext(filename)
            output_filename = f"region_{index}.png"
            output_path = os.path.join(output_folder, output_filename)
            cropped_image.save(output_path)
            print(f"Cropped image saved as: {output_path}")

if __name__ == "__main__":
    with open(r"D:\gpu-ocr\cropper\via_project_13Feb2025_21h39m_json.json", 'r') as f:
        json_data = json.load(f)
    process_json(json_data)