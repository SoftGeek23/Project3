from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

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

# Load model
model = AlternateLeNet5()
model.load_state_dict(torch.load("optimized_lenet5.pth", map_location="cpu"))
model.eval()

# Transform (no augmentation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class_labels = ["damage", "no_damage"]

@app.route("/summary", methods=["GET"])
def summary():
    return jsonify({
        "model": "Alternate LeNet-5",
        "input_size": [3, 128, 128],
        "output_classes": class_labels,
        "author": "Moksh Nirvaan"
    })

@app.route("/inference", methods=["POST"])
def inference():
    try:
        image_data = request.get_data()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        output = model(img_tensor)
        pred = output.argmax(1).item()
        return jsonify({"prediction": class_labels[pred]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
