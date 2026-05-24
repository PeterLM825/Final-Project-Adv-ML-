import os
import io
import base64
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# ── Model Architecture (mirrors notebook exactly) ─────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models as tv_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 90-class animal list derived from the dataset (alphabetical, matching LabelEncoder order)
CLASS_NAMES = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar",
    "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow",
    "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly",
    "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish",
    "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster",
    "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter",
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin",
    "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer",
    "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep",
    "snake", "sparrow", "squid", "squirrel", "starfish", "swan", "tiger",
    "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
]
NUM_CLASSES = len(CLASS_NAMES)  # 90

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

device = None
backbone = None
classifier = None
model_loaded = False

if TORCH_AVAILABLE:
    device = torch.device("cpu")

    class NormalizedResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            base = tv_models.resnet50(weights="IMAGENET1K_V2")
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        def forward(self, x):
            x = (x - self.mean) / self.std
            x = self.backbone(x)
            return x.squeeze(-1).squeeze(-1)

    class DeepFeatureClassifier(nn.Module):
        def __init__(self, in_features=2048, h1=1024, h2=512,
                     droprate=0.4, num_classes=90):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Dropout(droprate),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Dropout(droprate * 0.75),
                nn.Linear(h2, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    def load_models():
        global backbone, classifier, model_loaded
        try:
            backbone = NormalizedResNet50().to(device)
            backbone.eval()

            classifier = DeepFeatureClassifier(num_classes=NUM_CLASSES).to(device)
            model_path = os.path.join("models", "best_final_model_r50.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join("models", "animal_classifier.pt")
            state_dict = torch.load(model_path, map_location=device)
            classifier.load_state_dict(state_dict)
            classifier.eval()

            model_loaded = True
            print(f"Models loaded from {model_path}")
        except Exception as e:
            print(f"Model load error: {e}")
            model_loaded = False

    load_models()

    RAW_TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    def predict_image(pil_img, top_k=5):
        img_tensor = RAW_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat   = backbone(img_tensor)
            logits = classifier(feat)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [
            {"rank": i + 1,
             "label": CLASS_NAMES[idx],
             "confidence": float(probs[idx]) * 100}
            for i, idx in enumerate(top_indices)
        ]
        return results

else:
    model_loaded = False

    def predict_image(pil_img, top_k=5):
        raise RuntimeError("PyTorch is not installed. Cannot run inference.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_b64(pil_img, max_size=400):
    """Resize and base64-encode an image for inline display."""
    w, h = pil_img.size
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           model_loaded=model_loaded,
                           num_classes=NUM_CLASSES)


@app.route("/profile")
def profile():
    return render_template("profile.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, BMP, or WEBP."}), 400

    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 503

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results   = predict_image(pil_img, top_k=5)
        img_b64   = pil_to_b64(pil_img)
        return jsonify({"predictions": results, "image": img_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded,
                    "torch_available": TORCH_AVAILABLE,
                    "num_classes": NUM_CLASSES})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
