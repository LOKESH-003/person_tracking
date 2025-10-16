import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import sqlite3
import face_recognition

# ---------------- CONFIG ----------------
EMB_DB = "embeddings.db"
PEOPLE_IMAGES_DIR = "people_images"
MIN_BOX_AREA = 1600
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FEATURE EXTRACTOR ----------------
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_body_embedding(img_np):
    img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor)
        feats = feats.reshape(1,-1)
        feats = F.normalize(feats, p=2, dim=1)
        emb = feats.cpu().numpy().astype(np.float32).flatten()
    return emb

def extract_face_embedding(img_np):
    rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    face_boxes = face_recognition.face_locations(rgb, model="hog")
    if not face_boxes:
        return None
    encodings = face_recognition.face_encodings(rgb, face_boxes)
    if not encodings:
        return None
    emb = encodings[0].astype(np.float32)
    emb = emb / (np.linalg.norm(emb)+1e-12)
    return emb

def np_to_blob(arr):
    return arr.astype(np.float32).tobytes()

# ---------------- SQLITE ----------------
def init_db():
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS people_embeddings (
        name TEXT PRIMARY KEY,
        body_embedding BLOB,
        face_embedding BLOB
    )
    """)
    conn.commit()
    conn.close()

def insert_embedding(name, body_emb, face_emb):
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO people_embeddings(name, body_embedding, face_embedding) VALUES (?, ?, ?)",
              (name, np_to_blob(body_emb), np_to_blob(face_emb) if face_emb is not None else None))
    conn.commit()
    conn.close()

# ---------------- MAIN ----------------
def populate_db():
    init_db()
    for person_name in os.listdir(PEOPLE_IMAGES_DIR):
        person_dir = os.path.join(PEOPLE_IMAGES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        body_embeddings = []
        face_embeddings = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg','.png','.jpeg')):
                continue
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Body embedding
            body_emb = extract_body_embedding(img)
            if body_emb is not None:
                body_embeddings.append(body_emb)

            # Face embedding
            face_emb = extract_face_embedding(img)
            if face_emb is not None:
                face_embeddings.append(face_emb)

        if not body_embeddings:
            print(f"⚠️ No valid body embeddings for {person_name}, skipping...")
            continue

        # Average embeddings if multiple images
        avg_body = np.mean(np.stack(body_embeddings), axis=0)
        avg_body = avg_body / (np.linalg.norm(avg_body)+1e-12)

        avg_face = None
        if face_embeddings:
            avg_face = np.mean(np.stack(face_embeddings), axis=0)
            avg_face = avg_face / (np.linalg.norm(avg_face)+1e-12)

        insert_embedding(person_name, avg_body.astype(np.float32), avg_face.astype(np.float32) if avg_face is not None else None)
        print(f"✅ Added {person_name} to DB. Face: {'Yes' if avg_face is not None else 'No'}")

if __name__ == "__main__":
    populate_db()
