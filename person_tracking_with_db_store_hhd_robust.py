# file: yolo_person_persistent_tracking_v2.py
"""
Robust persistent person re-identification:
- ResNet50 feature extractor (pretrained)
- SQLite table `people_embeddings(name PRIMARY KEY, embedding BLOB)`
- Averaged embeddings per person with online update
- Cosine similarity matching (threshold adjustable)
"""

import os
import cv2
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import yt_dlp
from typing import List, Tuple, Dict, Optional

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11n.pt"                  # YOLO detector weights
INPUT_FILE = "youtube_input23.mp4"         # local video path
YOUTUBE_URL = "https://youtu.be/ulYDSTdbGa8?si=7KiFjKW3_A9HLKOv"
PEOPLE_DB = "people.db"                    # keeps the ordered list of names
EMB_DB = "embeddings.db"                   # persistent embeddings DB
OUTPUT_FILE = "output_with_names_v4test.mp4"
source=0 #"http://192.168.1.18:8080/video" 0
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.82     # cosine sim threshold to consider same person
UPDATE_ALPHA = 0.90        # new_avg = alpha*old + (1-alpha)*new
MIN_BOX_AREA = 1600        # ignore very small boxes (width*height)
BATCH_WRITE_INTERVAL = 500 # frames between flushes to DB

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FEATURE EXTRACTOR ----------------
# Use ResNet50 backbone; remove final classification head and take pooled features.
print("🔍 Loading ResNet50 feature extractor...")
resnet = models.resnet50(pretrained=True)
# Remove final fc layer -> get 2048-dim features from avgpool
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),  # tall crops typically person-like: (H, W)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_embedding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Crop bbox from frame and return L2-normalized embedding (float32) or None if invalid."""
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), max(0, min(frame.shape[1] - 1, x2))
    y1, y2 = max(0, y1), max(0, min(frame.shape[0] - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
        return None
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor)            # shape [1, 2048, 1, 1]
        feats = feats.reshape(1, -1)                 # [1, 2048]
        feats = F.normalize(feats, p=2, dim=1)      # L2 normalize
        emb = feats.cpu().numpy().astype(np.float32).flatten()
    return emb

# ---------------- SQLITE HELPERS ----------------
def init_db():
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS people_embeddings (
                 name TEXT PRIMARY KEY,
                 embedding BLOB
                 )""")
    conn.commit()
    conn.close()

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def load_all_embeddings() -> List[Tuple[str, np.ndarray]]:
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM people_embeddings")
    rows = c.fetchall()
    conn.close()
    data = []
    for name, blob in rows:
        try:
            emb = blob_to_np(blob)
            # safety: ensure 1D float32
            emb = emb.astype(np.float32).flatten()
            data.append((name, emb))
        except Exception:
            continue
    return data

def insert_embedding_to_db(name: str, emb: np.ndarray):
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO people_embeddings(name, embedding) VALUES (?, ?)",
              (name, np_to_blob(emb)))
    conn.commit()
    conn.close()

def bulk_update_embeddings(updates: Dict[str, np.ndarray]):
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    rows = [(name, np_to_blob(emb)) for name, emb in updates.items()]
    c.executemany("INSERT OR REPLACE INTO people_embeddings(name, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

# ---------------- UTIL ----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return -1.0
    # A and B are expected to be already normalized; but compute robustly
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def find_best_match(emb: np.ndarray, memory: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
    if emb is None or len(memory) == 0:
        return None, -1.0
    names = [n for n, e in memory]
    embs = np.stack([e for _, e in memory])        # shape [N, D]
    # if memory embeddings not normalized, normalize rows
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
    sims = float(np.max(np.dot(emb_norm, embs_norm.T)))
    idx = int(np.argmax(np.dot(emb_norm, embs_norm.T)))
    return names[idx], sims

# ---------------- VIDEO / YOLO SETUP ----------------
print(f"✅ Loading YOLO model to {device}...")
yolo = YOLO(MODEL_PATH)
yolo.to(device)

def download_youtube(url: str, save_as: str) -> str:
    if os.path.exists(save_as):
        print(f"🎬 Using cached: {save_as}")
        return save_as
    print("📥 Downloading video...")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': save_as, 'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("✅ Downloaded.")
    return save_as

def load_names_ordered() -> List[str]:
    if not os.path.exists(PEOPLE_DB):
        return []
    conn = sqlite3.connect(PEOPLE_DB)
    c = conn.cursor()
    try:
        c.execute("SELECT name FROM people ORDER BY id ASC")
        rows = c.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()

# ---------------- MAIN ----------------
def main():
    init_db()
    memory = load_all_embeddings()   # list[(name, emb_np)]
    name_list = load_names_ordered()
    print(f"📚 Loaded {len(memory)} stored embeddings and {len(name_list)} names from people.db")
    # map track_id -> assigned name
    assigned_names: Dict[int, str] = {}
    # in-memory updated embeddings before periodic flush
    pending_updates: Dict[str, np.ndarray] = {}
    next_name_idx = 0

    # ensure next_name_idx respects names already in DB if any
    # (if some names are already used by embeddings, don't reuse them)
    used_names = {n for n, _ in memory}
    # advance next_name_idx until we find an unused name
    while next_name_idx < len(name_list) and name_list[next_name_idx] in used_names:
        next_name_idx += 1
        
    # prepare video
    video_path = download_youtube(YOUTUBE_URL, INPUT_FILE)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Unable to open video")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_idx = 0
    print("🚀 Starting persistent person tracking (v2)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = yolo.track(frame, persist=True, verbose=False,
                                 conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            if results is None or len(results) == 0 or results[0].boxes is None:
                out.write(frame)
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            if ids is None:
                ids = [None] * len(boxes)

            for bbox, cls_idx, track_id in zip(boxes, classes, ids):
                if model_name := yolo.names.get(int(cls_idx), None) is None:
                    # In case yolo.names is not dict-like in this version:
                    pass
                # Only process people
                # Ultralytics typically uses class 0 for 'person', but we check by name
                cls_name = yolo.names[int(cls_idx)]
                if cls_name != "person" or track_id is None:
                    continue
                track_id = int(track_id)

                emb = extract_embedding(frame, bbox)
                if emb is None:
                    # couldn't extract a valid embedding; skip annotation
                    continue

                if track_id in assigned_names:
                    name = assigned_names[track_id]
                    # Update that person's embedding in pending_updates
                    # prefer updating memory's embedding if present
                    # find memory entry
                    found = False
                    for i, (n, e) in enumerate(memory):
                        if n == name:
                            new_avg = (UPDATE_ALPHA * e) + ((1.0 - UPDATE_ALPHA) * emb)
                            # renormalize
                            new_avg = new_avg / (np.linalg.norm(new_avg) + 1e-12)
                            memory[i] = (n, new_avg.astype(np.float32))
                            pending_updates[n] = new_avg.astype(np.float32)
                            found = True
                            break
                    if not found:
                        # Might be a newly assigned name that wasn't in memory yet
                        pending_updates[name] = emb.astype(np.float32)
                        memory.append((name, emb.astype(np.float32)))
                else:
                    # Attempt to reconnect via matching
                    match_name, sim = find_best_match(emb, memory)
                    if match_name is not None and sim >= MATCH_THRESHOLD:
                        name = match_name
                        # update averaged embedding
                        for i, (n, e) in enumerate(memory):
                            if n == name:
                                new_avg = (UPDATE_ALPHA * e) + ((1.0 - UPDATE_ALPHA) * emb)
                                new_avg = new_avg / (np.linalg.norm(new_avg) + 1e-12)
                                memory[i] = (n, new_avg.astype(np.float32))
                                pending_updates[n] = new_avg.astype(np.float32)
                                break
                        assigned_names[track_id] = name
                        # ensure next_name_idx still consistent
                        if name in used_names:
                            pass
                        print(f"🔁 Reconnected track {track_id} -> {name} (sim={sim:.3f})")
                    else:
                        # New person: assign next available name from people.db or Unknown
                        if next_name_idx < len(name_list):
                            name = name_list[next_name_idx]
                            next_name_idx += 1
                        else:
                            # create Unknown_N unique name
                            kne = 0
                            while True:
                                cand = f"Unknown_{kne}"
                                if cand not in used_names:
                                    name = cand
                                    break
                                kne += 1
                        assigned_names[track_id] = name
                        # persist immediately new embedding
                        emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
                        memory.append((name, emb_norm.astype(np.float32)))
                        pending_updates[name] = emb_norm.astype(np.float32)
                        used_names.add(name)
                        print(f"🆕 Assigned new name {name} to track {track_id}")

                # draw box + label
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                display_name = assigned_names.get(track_id, "...")
                cv2.putText(frame, display_name, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            out.write(frame)
            # show scaled preview
            preview = cv2.resize(frame, (640,640)) if frame.shape[0] >= 640 and frame.shape[1] >= 640 else frame
            # re=cv2.resize(preview,(1500,800))
            cv2.imshow("Persistent ReID v2", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Periodic flush to DB
            if frame_idx % BATCH_WRITE_INTERVAL == 0 and pending_updates:
                try:
                    bulk_update_embeddings(pending_updates)
                    pending_updates.clear()
                    # reload memory from DB to ensure consistency (optional)
                    memory = load_all_embeddings()
                    print(f"💾 Flushed embeddings to DB at frame {frame_idx}")
                except Exception as e:
                    print("⚠️ Error writing embeddings to DB:", e)

    finally:
        # final flush
        if pending_updates:
            try:
                bulk_update_embeddings(pending_updates)
                pending_updates.clear()
                print("💾 Final embeddings flush complete.")
            except Exception as e:
                print("⚠️ Final DB flush error:", e)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Done. Output saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
