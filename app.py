"""
Animal Classifier Flask Application
90-class animal image classification using ResNet-50 feature extraction.
Author: Peter | Advanced Machine Learning Final Project
"""

import os
import io
import json
import base64
import datetime
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models as tv_models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_from_directory, flash)
from werkzeug.utils import secure_filename

# ─────────────────────────── App Config ───────────────────────────────────────
app = Flask(__name__)
app.secret_key = "animal_classifier_secret_2024"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
VISITOR_FILE = 'static/data/visitors.json'

# ─────────────────────────── 90 Animal Classes ────────────────────────────────
CLASS_NAMES = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar',
    'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote',
    'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck',
    'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish',
    'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
    'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish',
    'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster',
    'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter',
    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin',
    'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer',
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
    'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey',
    'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

ANIMAL_FACTS = {
    'antelope': 'Antelopes can run up to 60 mph — faster than most predators.',
    'badger': 'Badgers are the world\'s most powerful diggers for their size.',
    'bat': 'Bats are the only mammals capable of sustained flight.',
    'bear': 'Bears have an exceptional sense of smell — 2,100 times better than humans.',
    'bee': 'A single bee can visit up to 5,000 flowers in one day.',
    'beetle': 'Beetles make up 25% of all animal species on Earth.',
    'bison': 'American bison can run at speeds of 35 mph despite their size.',
    'boar': 'Wild boars have an excellent memory and can remember routes they\'ve traveled.',
    'butterfly': 'Butterflies taste with their feet using chemoreceptors.',
    'cat': 'Cats spend 70% of their lives sleeping — up to 16 hours a day.',
    'caterpillar': 'Caterpillars completely dissolve into liquid during metamorphosis.',
    'chimpanzee': 'Chimpanzees share about 98.7% of their DNA with humans.',
    'cockroach': 'Cockroaches can survive up to a week without their head.',
    'cow': 'Cows have panoramic vision covering almost 360 degrees.',
    'coyote': 'Coyotes mate for life and are highly adaptable to urban environments.',
    'crab': 'Crabs have 10 legs and communicate by drumming or waving their claws.',
    'crow': 'Crows can recognize human faces and hold grudges for years.',
    'deer': 'Deer can jump 10 feet high and run up to 30 mph.',
    'dog': 'Dogs have a sense of smell 10,000–100,000 times stronger than humans.',
    'dolphin': 'Dolphins sleep with one eye open, resting only half their brain at a time.',
    'donkey': 'Donkeys have an excellent memory and can recognize places after 25 years.',
    'dragonfly': 'Dragonflies have a 95% hunting success rate — the highest of any predator.',
    'duck': 'Ducks have waterproof feathers thanks to a special gland near their tail.',
    'eagle': 'Eagles can spot a rabbit from nearly 2 miles away.',
    'elephant': 'Elephants are the only animals that can\'t jump — and they don\'t need to.',
    'flamingo': 'Flamingos are pink because of the carotenoid pigments in their food.',
    'fly': 'House flies have 4,000 lenses in each eye, giving them near-360 vision.',
    'fox': 'Foxes use Earth\'s magnetic field to hunt with remarkable accuracy.',
    'goat': 'Goats were one of the first animals domesticated by humans, ~10,000 years ago.',
    'goldfish': 'Goldfish have a memory span of at least 3 months — not 3 seconds.',
    'goose': 'Geese fly in V-formation to reduce wind resistance by up to 70%.',
    'gorilla': 'Gorillas share 98.3% of their DNA with humans.',
    'grasshopper': 'Grasshoppers can leap 20 times their body length in one jump.',
    'hamster': 'Hamsters can run up to 6 miles per night on their wheel.',
    'hare': 'Hares are born with open eyes and full fur — unlike rabbits.',
    'hedgehog': 'Hedgehogs are immune to many snake venoms.',
    'hippopotamus': 'Hippos secrete a natural sunscreen that also acts as an antibiotic.',
    'hornbill': 'Hornbills seal their nest entrance with mud to protect their eggs.',
    'horse': 'Horses can sleep both standing up and lying down.',
    'hummingbird': 'Hummingbirds are the only birds that can fly backwards.',
    'hyena': 'Hyenas are more closely related to cats than to dogs.',
    'jellyfish': 'One jellyfish species (Turritopsis dohrnii) is considered biologically immortal.',
    'kangaroo': 'Kangaroos can\'t walk backwards — their legs won\'t allow it.',
    'koala': 'Koalas sleep up to 22 hours a day to conserve energy digesting eucalyptus.',
    'ladybugs': 'Ladybugs can play dead to fool predators.',
    'leopard': 'Leopards are expert climbers and often haul prey into trees.',
    'lion': 'A lion\'s roar can be heard from 5 miles away.',
    'lizard': 'Some lizards can detach their tail as a defense mechanism.',
    'lobster': 'Lobsters taste with their legs and chew with their stomachs.',
    'mosquito': 'Mosquitoes are the deadliest animal on Earth, responsible for millions of deaths.',
    'moth': 'Moths navigate by the moon — which is why lights confuse them.',
    'mouse': 'Mice are social animals that laugh (ultrasonically) when tickled.',
    'octopus': 'Octopuses have three hearts, blue blood, and nine brains.',
    'okapi': 'Okapis are the only living relative of the giraffe.',
    'orangutan': 'Orangutans build a new sleeping nest in the tree canopy every single night.',
    'otter': 'Sea otters hold hands while sleeping so they don\'t drift apart.',
    'owl': 'Owls can rotate their heads 270 degrees.',
    'ox': 'Oxen were essential to early human civilization for plowing and transport.',
    'oyster': 'Oysters can change their gender multiple times throughout their lives.',
    'panda': 'Giant pandas spend 12-16 hours a day eating bamboo.',
    'parrot': 'Some parrots can live for over 80 years.',
    'pelecaniformes': 'Pelicans can hold 3 gallons of water in their pouch.',
    'penguin': 'Emperor penguins can dive to depths of 1,850 feet.',
    'pig': 'Pigs are smarter than dogs and can learn their name in 2 weeks.',
    'pigeon': 'Pigeons can recognize themselves in a mirror — a rare cognitive ability.',
    'porcupine': 'Porcupines have up to 30,000 quills on their body.',
    'possum': 'Opossums are immune to most snake venoms and rarely get rabies.',
    'raccoon': 'Raccoons can unlock doors and containers with their nimble hands.',
    'rat': 'Rats can laugh (ultrasonically) when playing and feel empathy for others.',
    'reindeer': 'Reindeer eyes change color from gold in summer to blue in winter.',
    'rhinoceros': 'Rhino horns are made of keratin — the same material as human fingernails.',
    'sandpiper': 'Sandpipers migrate thousands of miles without stopping to rest.',
    'seahorse': 'Male seahorses carry and give birth to their young.',
    'seal': 'Seals can hold their breath for up to 2 hours.',
    'shark': 'Sharks have existed for over 450 million years — older than trees.',
    'sheep': 'Sheep have excellent memories and can recognize up to 50 other sheep faces.',
    'snake': 'Snakes smell with their tongues, flicking them to collect scent particles.',
    'sparrow': 'House sparrows were introduced to New York in 1851 and now span North America.',
    'squid': 'Giant squids have the largest eyes in the animal kingdom.',
    'squirrel': 'Squirrels plant thousands of trees by forgetting where they buried nuts.',
    'starfish': 'Starfish have no brain or blood — they use seawater for circulation.',
    'swan': 'Swans mate for life and can live up to 20 years in the wild.',
    'tiger': 'Tigers are the largest wild cat and excellent swimmers.',
    'turkey': 'Wild turkeys can run at 25 mph and fly short distances at 55 mph.',
    'turtle': 'Some turtles can breathe through their butts (via cloacal bursae).',
    'whale': 'Blue whales have hearts the size of a small car.',
    'wolf': 'Wolves howl to communicate across distances up to 10 miles.',
    'wombat': 'Wombats produce cube-shaped droppings — unique in the animal kingdom.',
    'woodpecker': 'Woodpeckers peck up to 20 times per second without getting headaches.',
    'zebra': 'Each zebra\'s stripe pattern is unique — like a human fingerprint.',
}

# ─────────────────────────── Model Architecture ───────────────────────────────
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(imagenet_mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(imagenet_std).view(1,3,1,1))
        backbone = tv_models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

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

# ─────────────────────────── Global model state ───────────────────────────────
backbone_model = None
classifier_model = None
baseline_model = None
model_loaded = False

raw_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_models():
    """Load models from disk if available."""
    global backbone_model, classifier_model, baseline_model, model_loaded
    try:
        backbone_model = NormalizedResNet50().to(device)
        backbone_model.eval()

        classifier_model = DeepFeatureClassifier(num_classes=90).to(device)
        if os.path.exists('animal_classifier.pt'):
            state = torch.load('animal_classifier.pt', map_location=device)
            classifier_model.load_state_dict(state)
            classifier_model.eval()
            model_loaded = True
            print("✓ Final model loaded from animal_classifier.pt")
        elif os.path.exists('best_final_model_r50.pt'):
            state = torch.load('best_final_model_r50.pt', map_location=device)
            classifier_model.load_state_dict(state)
            classifier_model.eval()
            model_loaded = True
            print("✓ Final model loaded from best_final_model_r50.pt")

        baseline_model = DeepFeatureClassifier(num_classes=90).to(device)
        if os.path.exists('best_baseline.pt'):
            state = torch.load('best_baseline.pt', map_location=device)
            baseline_model.load_state_dict(state)
            baseline_model.eval()
            print("✓ Baseline model loaded from best_baseline.pt")

    except Exception as e:
        print(f"⚠ Model loading error: {e}")
        print("  Running in DEMO mode — predictions are simulated.")
        model_loaded = False

def classify_single(image_path):
    """Extract features and classify a single image."""
    if not model_loaded:
        # Demo mode: return plausible mock predictions
        top_idx = np.random.choice(len(CLASS_NAMES), 5, replace=False)
        raw = np.random.dirichlet(np.ones(5) * 0.5)
        raw = np.sort(raw)[::-1]
        return [(CLASS_NAMES[i], float(p)) for i, p in zip(top_idx, raw)]

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = raw_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat   = backbone_model(img_tensor)
        logits = classifier_model(feat)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx  = np.argsort(probs)[::-1][:5]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]

def classify_single_baseline(image_path):
    """Run baseline model on a single image."""
    if not model_loaded or baseline_model is None:
        top_idx = np.random.choice(len(CLASS_NAMES), 5, replace=False)
        raw = np.random.dirichlet(np.ones(5) * 0.5)
        raw = np.sort(raw)[::-1]
        return [(CLASS_NAMES[i], float(p)) for i, p in zip(top_idx, raw)]

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = raw_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat   = backbone_model(img_tensor)
        logits = baseline_model(feat)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx  = np.argsort(probs)[::-1][:5]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]

def get_similar_images(predicted_class, n=3):
    """
    Find similar images from the dataset folder for the predicted class.
    Returns list of relative paths (relative to static/).
    """
    dataset_root = os.environ.get('DATASET_PATH', 'animals/animals')
    class_dir = os.path.join(dataset_root, predicted_class)
    results = []
    if os.path.isdir(class_dir):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(exts)]
        chosen = np.random.choice(files, min(n, len(files)), replace=False) if files else []
        for fname in chosen:
            src = os.path.join(class_dir, fname)
            dest_name = f"similar_{predicted_class}_{fname}"
            dest = os.path.join('static/uploads', dest_name)
            if not os.path.exists(dest):
                try:
                    img = Image.open(src).convert('RGB')
                    img.thumbnail((300, 300))
                    img.save(dest)
                except Exception:
                    continue
            results.append(f"uploads/{dest_name}")
    return results

# ─────────────────────────── Visitor Counter ──────────────────────────────────
def get_visitors():
    os.makedirs(os.path.dirname(VISITOR_FILE), exist_ok=True)
    if os.path.exists(VISITOR_FILE):
        with open(VISITOR_FILE) as f:
            return json.load(f).get('count', 0)
    return 0

def increment_visitors():
    count = get_visitors() + 1
    with open(VISITOR_FILE, 'w') as f:
        json.dump({'count': count}, f)
    return count

# ─────────────────────────── Utility Helpers ──────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_top5_chart(predictions):
    """Return base64-encoded top-5 bar chart PNG."""
    labels = [p[0].replace('_', ' ').title() for p in predictions]
    probs  = [p[1] * 100 for p in predictions]
    colors = []
    for i, p in enumerate(probs):
        if i == 0:
            colors.append('#10b981' if p >= 50 else '#f59e0b' if p >= 25 else '#ef4444')
        else:
            colors.append('#94a3b8')

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    bars = ax.barh(labels[::-1], probs[::-1], color=colors[::-1],
                   edgecolor='none', height=0.6)
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%', va='center', color='#e2e8f0', fontsize=10,
                fontweight='bold')
    ax.set_xlim(0, max(probs) * 1.2)
    ax.set_xlabel('Confidence (%)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_comparison_chart(final_preds, baseline_preds):
    """Side-by-side comparison chart of two models."""
    labels = list(set([p[0] for p in final_preds[:3]] + [p[0] for p in baseline_preds[:3]]))[:5]
    final_d    = {p[0]: p[1]*100 for p in final_preds}
    baseline_d = {p[0]: p[1]*100 for p in baseline_preds}
    f_vals = [final_d.get(l, 0) for l in labels]
    b_vals = [baseline_d.get(l, 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.bar(x - width/2, b_vals, width, label='Baseline', color='#64748b', edgecolor='none')
    ax.bar(x + width/2, f_vals, width, label='Final Model', color='#6366f1', edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_',' ').title() for l in labels],
                       rotation=20, ha='right', color='#cbd5e1', fontsize=8)
    ax.set_ylabel('Confidence (%)', color='#94a3b8')
    ax.tick_params(colors='#cbd5e1')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0', edgecolor='#334155')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────── Routes ───────────────────────────────────────────
@app.before_request
def track_visitor():
    if 'visited' not in session:
        session['visited'] = True
        increment_visitors()
    if 'history' not in session:
        session['history'] = []

@app.route('/')
def index():
    visitors = get_visitors()
    return render_template('index.html', visitors=visitors,
                           total_classes=len(CLASS_NAMES))

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        flash('No image file selected.', 'error')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please upload a valid image file (JPG, PNG, BMP, WEBP).', 'error')
        return redirect(url_for('index'))

    fname = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(save_path)

    predictions = classify_single(save_path)
    top_class, top_conf = predictions[0]
    chart_b64 = make_top5_chart(predictions)
    similar   = get_similar_images(top_class, n=3)
    fact      = ANIMAL_FACTS.get(top_class, f"The {top_class} is one of the 90 species in this dataset.")

    conf_level = 'high' if top_conf >= 0.50 else 'medium' if top_conf >= 0.25 else 'low'
    warning    = None
    if top_conf < 0.40:
        warning = "Model confidence is below 40% — try a clearer or closer image."

    # Append to session history
    history = session.get('history', [])
    history.insert(0, {
        'filename': file.filename,
        'saved':    fname,
        'class':    top_class,
        'conf':     round(top_conf * 100, 1),
        'time':     datetime.datetime.now().strftime('%H:%M:%S'),
    })
    session['history'] = history[:20]
    session.modified = True

    return render_template('result.html',
        uploaded=f"uploads/{fname}",
        predictions=predictions,
        top_class=top_class,
        top_conf=round(top_conf * 100, 1),
        conf_level=conf_level,
        warning=warning,
        chart_b64=chart_b64,
        similar=similar,
        fact=fact,
        visitors=get_visitors())

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('images')
        for file in files:
            if file and allowed_file(file.filename):
                fname = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                file.save(save_path)
                preds = classify_single(save_path)
                top_class, top_conf = preds[0]
                results.append({
                    'filename': file.filename,
                    'saved':    fname,
                    'class':    top_class,
                    'conf':     round(top_conf * 100, 1),
                    'conf_level': 'high' if top_conf >= 0.50 else 'medium' if top_conf >= 0.25 else 'low',
                    'top5':     [(p[0], round(p[1]*100,1)) for p in preds],
                })
    return render_template('batch.html', results=results, visitors=get_visitors())

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            fname = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(save_path)
            final_preds    = classify_single(save_path)
            baseline_preds = classify_single_baseline(save_path)
            chart_b64      = make_comparison_chart(final_preds, baseline_preds)
            result = {
                'uploaded':       f"uploads/{fname}",
                'final_preds':    [(p[0], round(p[1]*100,1)) for p in final_preds],
                'baseline_preds': [(p[0], round(p[1]*100,1)) for p in baseline_preds],
                'chart_b64':      chart_b64,
                'final_class':    final_preds[0][0],
                'baseline_class': baseline_preds[0][0],
            }
    return render_template('compare.html', result=result, visitors=get_visitors())

@app.route('/history')
def history():
    h = session.get('history', [])
    return render_template('history.html', history=h, visitors=get_visitors())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    session.modified = True
    return redirect(url_for('history'))

@app.route('/explorer')
def explorer():
    return render_template('explorer.html',
                           class_names=CLASS_NAMES,
                           total=len(CLASS_NAMES),
                           facts_json=json.dumps(ANIMAL_FACTS),
                           visitors=get_visitors())

@app.route('/about')
def about():
    return render_template('about.html', visitors=get_visitors())

@app.route('/api/visitors')
def api_visitors():
    return jsonify({'count': get_visitors()})

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/data', exist_ok=True)
    load_models()
    app.run(debug=True, port=5000)
