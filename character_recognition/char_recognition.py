import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class CharacterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CharacterClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def create_data_loaders(data_dir, batch_size, validation_split):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Calculate lengths for split
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, len(dataset.classes)

def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    
    # Plot training metrics
    train_mean = np.array([x['mean'] for x in train_metrics])
    train_std = np.array([x['std'] for x in train_metrics])
    plt.plot(train_mean, label='Training')
    plt.fill_between(
        range(len(train_mean)),
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2
    )
    
    # Plot validation metrics
    val_mean = np.array([x['mean'] for x in val_metrics])
    val_std = np.array([x['std'] for x in val_metrics])
    plt.plot(val_mean, label='Validation')
    plt.fill_between(
        range(len(val_mean)),
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2
    )
    
    plt.title(f'{metric_name} over Training')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model2(data_dir, model_save_path, num_epochs=50, batch_size=32,
                learning_rate=0.001, patience=7, validation_split=0.2):
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Setup data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        data_dir, batch_size, validation_split
    )
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharacterClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = []
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{np.mean(train_loss):.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = []
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{np.mean(val_loss):.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Calculate epoch metrics
        epoch_train_loss = {'mean': np.mean(train_loss), 'std': np.std(train_loss)}
        epoch_train_acc = {'mean': train_correct / train_total, 'std': 0}
        epoch_val_loss = {'mean': np.mean(val_loss), 'std': np.std(val_loss)}
        epoch_val_acc = {'mean': val_correct / val_total, 'std': 0}
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Calculate additional metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Train Loss: {epoch_train_loss["mean"]:.4f} ± {epoch_train_loss["std"]:.4f}')
        print(f'Train Accuracy: {100 * epoch_train_acc["mean"]:.2f}%')
        print(f'Val Loss: {epoch_val_loss["mean"]:.4f} ± {epoch_val_loss["std"]:.4f}')
        print(f'Val Accuracy: {100 * epoch_val_acc["mean"]:.2f}%')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        
        # Check early stopping
        early_stopping(epoch_val_loss['mean'])
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered after epoch {epoch+1}')
            break
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, 'Loss', 'visualizations/loss_curves.png')
    plot_metrics(train_accuracies, val_accuracies, 'Accuracy', 'visualizations/accuracy_curves.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, 
                         classes=range(num_classes),
                         save_path='visualizations/confusion_matrix.png')
    
    return model

def train_model(data_dir, model_save_path, num_epochs=100, batch_size=16,
                learning_rate=0.0005, patience=15, validation_split=0.2):
    # Setup data loaders with augmentation
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_val = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # Create datasets with different transforms
    full_dataset = datasets.ImageFolder(data_dir, transform_train)
    
    # Calculate splits
    val_size = int(validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override validation set transform
    val_dataset.dataset.transform = transform_val
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharacterClassifier(len(full_dataset.classes)).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Rest of the training loop remains the same...
    
    # Save best model instead of last model
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    train_model(
        data_dir="output/char_images",
        model_save_path="models/char_classifier.pth",
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=7,
        validation_split=0.2
    )