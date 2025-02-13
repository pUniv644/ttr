import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image, ImageOps, ImageEnhance
import os
from tqdm import tqdm
from model import SimpleCharClassifier

class CharacterDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = {'S': 0, '1': 1, '3': 2, '5': 3, 'A': 4}
        self.samples = []
        
        # Load all images
        for class_name in self.class_to_idx:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(data_dir="data/char_images", model_save_path="models/simple_char_classifier.pth"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Enhanced transforms with fixed size
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # First resize to target size
        transforms.RandomRotation(15),  # Random rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random translation
            scale=(0.8, 1.2),      # Random scaling
            shear=10               # Random shear
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = CharacterDataset(data_dir, transform=train_transform)
    val_dataset = CharacterDataset(data_dir, transform=val_transform)
    
    print(f"Found {len(train_dataset)} total images")
    print("Class distribution:")
    class_counts = {}
    for _, label in train_dataset.samples:
        label_name = list(train_dataset.class_to_idx.keys())[list(train_dataset.class_to_idx.values()).index(label)]
        class_counts[label_name] = class_counts.get(label_name, 0) + 1
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    # Calculate class weights for imbalanced dataset
    class_weights = []
    total_samples = len(train_dataset)
    for class_name in ['S', '1', '3', '5', 'A']:
        count = class_counts.get(class_name, 0)
        weight = total_samples / (len(class_counts) * count) if count > 0 else 1.0
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights)
    print("\nClass weights:", {name: f"{weight:.2f}" for name, weight in zip(['S', '1', '3', '5', 'A'], class_weights)})
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = SimpleCharClassifier().to(device)
    class_weights = class_weights.to(device)
    
    # Use weighted loss for imbalanced dataset
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    try:
        for epoch in range(100):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_accuracy = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            class_correct = {i: 0 for i in range(5)}
            class_total = {i: 0 for i in range(5)}
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]'):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Per-class accuracy
                    for c in range(5):
                        mask = labels == c
                        class_total[c] += mask.sum().item()
                        class_correct[c] += (predicted[mask] == c).sum().item()
            
            val_accuracy = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # Print detailed results
            print(f'\nEpoch {epoch+1} Results:')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
            print('\nPer-class Validation Accuracy:')
            for c, label in enumerate(['S', '1', '3', '5', 'A']):
                if class_total[c] > 0:
                    print(f'{label}: {100. * class_correct[c] / class_total[c]:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                print(f"Saving best model (accuracy: {val_accuracy:.2f}%)")
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after epoch {epoch+1}')
                    break
        
        print("Training completed!")
        return model
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return model
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise  # Re-raise the exception for debugging
        return None

if __name__ == "__main__":
    # Training settings
    data_dir = "data/char_images"  # Directory with class subdirectories
    model_save_path = "models/simple_char_classifier.pth"
    
    # Start training
    model = train_model(data_dir, model_save_path)