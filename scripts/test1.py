# scripts/train_emotion_recognition.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress warnings (optional)
import warnings
warnings.filterwarnings("ignore")

# Define Constants
DATASET_NAME = "tukey/human_face_emotions_roboflow"
SAMPLE_SIZE = 5000          # Set to 3000 to sample 3000 samples
BATCH_SIZE = 8              # Reduced from 32 to fit GPU memory
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "../models/emotion_model_best34.pth"  # Relative path from 'scripts/'
VALIDATION_SPLIT = 0.2      # 20% for validation
SEED = 42                   # For reproducibility

# Set random seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Enable CuDNN Benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Load the dataset
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME)

    # Access the training split
    train_ds = ds['train']

    # Sample the dataset if SAMPLE_SIZE is set
    if SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} samples from the dataset...")
        sampled_train_ds = train_ds.select(range(SAMPLE_SIZE))
    else:
        sampled_train_ds = train_ds
        print("Using the entire dataset.")

    print(f"Sampled dataset size: {len(sampled_train_ds)}")

    # Extract images and labels
    print("Extracting images and labels...")
    images = []
    labels = []

    for sample in sampled_train_ds:
        image = sample['image']
        qa = sample['qa']
        if isinstance(qa, list) and len(qa) > 0:
            emotion = qa[0]['answer']
            images.append(image)
            labels.append(emotion)
        else:
            # Handle cases where 'qa' is not as expected
            images.append(image)
            labels.append("Neutral")  # Default to 'Neutral' if label is missing

    print(f"Total samples: {len(images)}")
    print(f"Sample labels: {labels[:5]}")

    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    print(f"Encoded labels: {encoded_labels[:5]}")
    print(f"Classes: {le.classes_}")

    # Shuffle the dataset
    print("Shuffling the dataset...")
    combined = list(zip(images, encoded_labels))
    random.shuffle(combined)
    images[:], encoded_labels[:] = zip(*combined)

    # Split into training and validation sets
    print("Splitting dataset into training and validation sets...")
    split_idx = int(len(images) * (1 - VALIDATION_SPLIT))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = encoded_labels[:split_idx], encoded_labels[split_idx:]

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    # Create the datasets
    train_dataset = EmotionDataset(train_images, train_labels, transform=transform)
    val_dataset = EmotionDataset(val_images, val_labels, transform=transform)

    # Create DataLoaders with optimized settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # Set to 0 to prevent memory overhead on Windows
        pin_memory=True if device.type == "cuda" else False  # Enable pin_memory for faster GPU transfer
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,      # Set to 0 to prevent memory overhead on Windows
        pin_memory=True if device.type == "cuda" else False  # Enable pin_memory for faster GPU transfer
    )

    # Clear GPU cache before loading the model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"Allocated GPU memory before loading model: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

    # Load pre-trained ResNet-18 model
    print("Loading pre-trained ResNet-18 model...")
    model = models.resnet34(pretrained=True)

    # Freeze early layers to save memory and computation
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer to match the number of emotion classes
    num_ftrs = model.fc.in_features
    num_classes = len(le.classes_)  # Number of emotion classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move the model to the appropriate device
    model = model.to(device)
    if device.type == "cuda":
        print(f"Allocated GPU memory after loading model: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Initialize variables to track the best model
    best_val_acc = 0.0

    # Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start_time = time.time()

        # Using tqdm for progress bar
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimization with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s")

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    val_outputs = model(inputs)
                    val_loss = criterion(val_outputs, labels)

                val_running_loss += val_loss.item() * inputs.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

        # Save the model if validation accuracy improves
        if val_epoch_acc > best_val_acc:
            print("Validation accuracy improved, saving model...")
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Training completed.")

if __name__ == "__main__":
    main()
