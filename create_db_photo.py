# file: preload_embeddings_db.py
import os
import cv2
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import face_recognition

DB_PATH = "embeddings_v3.db"
KNOWN_FACES_DIR = "./known_faces"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_BOX_AREA = 1600

# ---------------- FEATURE EXTRACTOR ----------------
print("🔍 Loading ResNet50 feature extractor (body)...")
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval().to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_body_embedding(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    if h * w < MIN_BOX_AREA:
        return None
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = feature_extractor(tensor).reshape(1, -1)
        feats = F.normalize(feats, p=2, dim=1)
    return feats.cpu().numpy().flatten().astype(np.float32)

# ---------------- SQLITE ----------------
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS people_embeddings (
            name TEXT PRIMARY KEY,
            body_embedding BLOB,
            face_embedding BLOB,
            photo BLOB
        )
    """)
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}.")

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes() if arr is not None else None

def insert_person(name, body_embedding=None, face_embedding=None, photo_bytes=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO people_embeddings
        (name, body_embedding, face_embedding, photo)
        VALUES (?, ?, ?, ?)
    """, (name, np_to_blob(body_embedding), np_to_blob(face_embedding), photo_bytes))
    conn.commit()
    conn.close()
    print(f"💾 Inserted/Updated: {name}")

# ---------------- MAIN ----------------
def main():
    init_database()

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"❌ Folder {KNOWN_FACES_DIR} not found.")
        return

    for file in os.listdir(KNOWN_FACES_DIR):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        name = os.path.splitext(file)[0]
        path = os.path.join(KNOWN_FACES_DIR, file)
        image = cv2.imread(path)
        if image is None:
            print(f"⚠️ Failed to read {path}")
            continue

        # Face embedding
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            face_encodings = face_recognition.face_encodings(image, face_locations)
            face_emb = face_encodings[0].astype(np.float32)
        else:
            face_emb = None
            print(f"⚠️ No face found for {name}")

        # Body embedding
        body_emb = extract_body_embedding(image)

        # Photo bytes
        _, jpeg = cv2.imencode('.jpg', image)
        photo_bytes = jpeg.tobytes()

        insert_person(name, body_emb, face_emb, photo_bytes)

if __name__ == "__main__":
    main()
