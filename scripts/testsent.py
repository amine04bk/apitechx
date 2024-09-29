import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
model = torch.load('emotion_model.pth')
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load and preprocess image
img = Image.open('face1.jpg')
img = transform(img).unsqueeze(0)  # Add batch dimension

# Predict emotion
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(f"Predicted Emotion: {emotions[predicted.item()]}")
