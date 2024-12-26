from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import torch
from torchvision import transforms
import base64
import numpy as np
from io import BytesIO
import torch.nn as nn

app = Flask(__name__)

# Define the Generator class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

# Load your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator()
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator = generator.to(device)

# Define the transformation to apply to the image before passing it through the model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def cartoonify_with_generator(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cartoonified_image = generator(image_tensor).cpu().squeeze(0)
    
    # Convert tensor to PIL image
    cartoonified_image = (cartoonified_image * 0.5 + 0.5) * 255
    cartoonified_image = cartoonified_image.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(cartoonified_image)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file:
            image = Image.open(file).convert("RGB")
            
            # Process image using the Generator model
            cartoon_image = cartoonify_with_generator(image)

            # Convert images to base64 for returning in the response
            buffered = BytesIO()
            cartoon_image.save(buffered, format="PNG")
            cartoonized_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                "cartoonized": cartoonized_image
            })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)