# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io, json, os, re
import pandas as pd
from collections import defaultdict
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

# ---------------- Config ----------------
MODEL_PATH = "model/model.h5"
LABELS_PATH = "model/labels.json"
IMAGE_SIZE = 224

# Custom objects for ViT (.h5)
custom_objects = {
    "ClassToken": ClassToken,
    "AddPositionEmbs": AddPositionEmbs,
    "TransformerBlock": TransformerBlock,
}

# --------------- App setup --------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------- Helpers ----------------
def norm(s: str) -> str:
    """Normalize disease/supplement names so joins succeed."""
    if s is None:
        return ""
    s = str(s)
    # unify smart punctuation, trim, lowercase
    s = s.replace("’", "'").replace("–", "-").replace("—", "-").strip().lower()
    # collapse spaces/underscores
    s = re.sub(r"[_\s]+", " ", s)
    # remove punctuation like : ( ) , / .
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# ------------- Load model/labels --------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects, safe_mode=False)

with open(LABELS_PATH) as f:
    labels = json.load(f)

# ------------- Load CSV data ------------
# Use a tolerant encoding for safety; change to "utf-8" if you resave the CSVs as UTF-8.
disease_df = pd.read_csv("data/disease_info.csv", encoding="latin1")
supp_df    = pd.read_csv("data/supplement_info.csv", encoding="latin1")

# Normalize keys for fast lookup
disease_df["__key"] = disease_df["disease_name"].apply(norm)
supp_df["__key"]    = supp_df["disease_name"].apply(norm)

DISEASES = {
    row["__key"]: {
        "disease_name": row.get("disease_name", ""),
        "description": row.get("description", "") or row.get("Description", ""),
        "possible_steps": row.get("Possible Steps", "") or row.get("possible_steps", ""),
        "image_url": row.get("image_url", "") or row.get("Image_URL", "")
    }
    for _, row in disease_df.iterrows()
}

SUPPLEMENTS = defaultdict(list)
for _, r in supp_df.iterrows():
    SUPPLEMENTS[r["__key"]].append({
        "supplement_name": r.get("supplement name", "") or r.get("supplement_name", ""),
        "supplement_image": r.get("supplement image", "") or r.get("supplement_image", ""),
        "buy_link": r.get("buy link", "") or r.get("buy_link", ""),
    })

SUPP_KEYS = set(SUPPLEMENTS.keys())

def find_best_disease_and_supplements(plant_class: str, disease_name_raw: str):
    """
    Try multiple normalized keys to join both DISEASES and SUPPLEMENTS.
    We try: disease only, plant+disease, disease+plant; then fuzzy contains.
    """
    # candidate normalized keys
    candidates = [
        norm(disease_name_raw),                        # e.g., "late blight"
        norm(f"{plant_class} {disease_name_raw}"),    # e.g., "potato late blight"
        norm(f"{disease_name_raw} {plant_class}"),    # e.g., "late blight potato"
    ]

    disease_rec = None
    supp_list = []

    # 1) exact matches
    for k in candidates:
        if not disease_rec:
            disease_rec = DISEASES.get(k)
        if not supp_list:
            supp_list = SUPPLEMENTS.get(k, [])
        if disease_rec and supp_list:
            break

    # 2) fuzzy contains for diseases
    if not disease_rec:
        for dk in DISEASES.keys():
            if any(k and (k in dk or dk in k) for k in candidates):
                disease_rec = DISEASES[dk]
                if not supp_list:
                    supp_list = SUPPLEMENTS.get(dk, [])
                break

    # 3) fuzzy contains for supplements (if still empty)
    if not supp_list:
        for sk in SUPP_KEYS:
            if any(k and (k in sk or sk in k) for k in candidates):
                supp_list = SUPPLEMENTS.get(sk, [])
                if supp_list:
                    break

    return disease_rec, supp_list

# Optional: debug a few keys
# print("Disease keys sample:", list(DISEASES.keys())[:10], flush=True)

# --------------- Routes -----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# Your existing simple endpoint (kept for compatibility)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    x = preprocess_image(contents)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = labels[idx]
    confidence = float(probs[idx])

    parts = label.split("___")
    plant_class = parts[0] if parts else "Unknown"
    disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    # (no CSV join here; frontend calling /predict_detail is recommended)
    return {"label": label, "plant_class": plant_class, "disease_name": disease_name, "confidence": confidence}

# Rich endpoint that joins model output to CSV info
@app.post("/predict_detail")
async def predict_detail(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        x = preprocess_image(contents)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = labels[idx]
        conf = float(probs[idx])

        parts = label.split("___")
        plant_class = parts[0] if parts else "Unknown"
        disease_name_raw = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        # normalized key for lookup
        disease_rec, supp_list = find_best_disease_and_supplements(plant_class, disease_name_raw)

        # 3) last resort: empty info but keep parsed disease name
        if disease_rec is None:
            disease_rec = {
                "disease_name": disease_name_raw,
                "description": "",
                "possible_steps": "",
                "image_url": ""
            }

        return {
            "plant_class": plant_class,
            "disease_name": disease_rec.get("disease_name") or disease_name_raw,
            "confidence": conf,
            "disease_info": {
                "description": disease_rec.get("description", ""),
                "possible_steps": disease_rec.get("possible_steps", ""),
                "image_url": disease_rec.get("image_url", "")
            },
            "supplements": supp_list
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
