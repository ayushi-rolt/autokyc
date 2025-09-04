


import torch
from PIL import Image
from torchvision import transforms
from .mobilefacenet import MobileFaceNet  # Ensure this import matches your project structure

# Load model
model = MobileFaceNet()
model = torch.jit.load("video_api/mobilefacenet_scripted.pt")  # Use relative path if possible
model.eval()

def get_face_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze(0)
