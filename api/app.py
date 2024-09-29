from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Constants
MODEL_SAVE_PATH = "../models/emotion_model_best11.pth"  # Path to your saved model
NUM_CLASSES = 8  # Adjust based on your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    model = models.resnet34(weights=None)  # Use weights=None for the updated version
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Read and process the image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Map the predicted class index to the emotion label (adjust based on your label encoding)
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # Adjust as needed
    predicted_emotion = emotion_labels[predicted.item()]

    return jsonify({"emotion": predicted_emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
