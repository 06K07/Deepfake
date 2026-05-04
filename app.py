from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
from werkzeug.utils import secure_filename

# ---------------- CONFIG ---------------- #
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- DEVICE ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using:", device)

if torch.cuda.is_available():
    print("🔥 GPU:", torch.cuda.get_device_name(0))


# ---------------- MODEL ---------------- #
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained model
MODEL_PATH = "best_resnet_model.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found! Run train.py first.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


# ---------------- TRANSFORM ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------- PREDICT ---------------- #
def predict_image(path):
    try:
        img = Image.open(path).convert("RGB")
    except:
        return "Invalid Image", 0.0

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"
    confidence = float(probs[pred])

    return label, confidence


# ---------------- ROUTES ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            result, confidence = predict_image(path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        filename=filename
    )


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    app.run(debug=True)