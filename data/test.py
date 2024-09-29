import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Define emotions and class mapping
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# Custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self.load_labels(label_file)

    def load_labels(self, label_file):
        """Load labels from a CSV file and return valid image-label pairs."""
        data = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    emotion = row[0].strip()  # Column 0: Emotion
                    img_name = row[1].strip().replace('"', '')  # Column 1: Image name
                    img_name = img_name.replace(";", "")  # Remove any accidental semicolons

                    image_path = os.path.join(self.img_dir, img_name)
                    
                    # Check if the image file exists and if the emotion is valid
                    if os.path.isfile(image_path) and emotion in emotion_to_idx:
                        data.append((image_path, emotion_to_idx[emotion]))
                    else:
                        print(f"Warning: {image_path} not found or invalid emotion '{emotion}'. Skipping this sample.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, emotion_idx = self.data[idx]
        image = Image.open(image_path)  # Do NOT convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(emotion_idx, dtype=torch.long)

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=15):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def main():
    # Define paths
    img_dir = './'  # All images are in the same folder
    label_file = './trainmanuelle.csv'  # CSV file is in the same folder

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match model input size
        transforms.ToTensor(),           # Convert images to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize for pre-trained models
    ])

    # Load dataset
    dataset = CustomDataset(img_dir, label_file, transform)

    # Ensure the dataset is not empty
    if len(dataset) == 0:
        print("Error: No valid samples found in the dataset. Exiting...")
        return

    # Create DataLoader for training
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained ResNet18 model
    model = models.resnet34(pretrained=True)

    # Modify the final layer to match the number of classes (7 emotions)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EMOTIONS))  # Output 7 classes

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=15)

    # Save the trained model
    torch.save(model.state_dict(), "emotion_model_bestalaa.pth")
    print("Training complete. Model saved as emotion_model_best.pth")

if __name__ == "__main__":
    main()
