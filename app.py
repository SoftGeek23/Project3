from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import traceback

app = Flask(__name__)

# Define the model
class AlternateLeNet5(nn.Module):
    def __init__(self):
        super(AlternateLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16 * 32 * 32, 120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load trained model
model = AlternateLeNet5()
model.load_state_dict(torch.load("optimized_lenet5.pth", map_location="cpu"))
model.eval()

# Transform definition
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class_labels = ["damage", "no_damage"]

# Summary endpoint
@app.route("/summary", methods=["GET"])
def summary():
    return jsonify({
        "model": "Alternate LeNet-5",
        "input_size": [3, 128, 128],
        "output_classes": class_labels,
        "author": "Moksh Nirvaan"
    })

# Inference endpoint
@app.route("/inference", methods=["POST"])
def inference():
    try:
        if 'image' not in request.files:
            print("❌ 'image' not in request.files")
            return jsonify({"error": "Missing image file"}), 400

        file = request.files['image']
        image_bytes = file.read()

        print("📥 Received image file:", file.filename)
        print("📏 Image byte size:", len(image_bytes))

        # Try OpenCV decoding
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is not None:
            print("✅ OpenCV decoded:", img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
        else:
            print("⚠️ OpenCV failed. Trying PIL...")
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                print("✅ PIL decoded image:", image.size)
            except Exception as e:
                print("❌ PIL failed:", e)
                return jsonify({"error": "invalid image"}), 400

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(1).item()

        return jsonify({"prediction": class_labels[pred]})

    except Exception as e:
        print("🔥 Inference error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
