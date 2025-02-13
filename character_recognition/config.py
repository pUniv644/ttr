import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR,
                os.path.join(OUTPUT_DIR, 'char_images'),
                os.path.join(OUTPUT_DIR, 'visualization')]:
    os.makedirs(dir_path, exist_ok=True)

# Model settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 36  # 10 digits + 26 letters

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Image settings
IMAGE_SIZE = 224  # ResNet default input size