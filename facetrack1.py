# file: yolo_person_persistent_tracking_v3.py
"""
Robust persistent multi-person re-identification (v3):
- ResNet50 feature extractor (pretrained)
- SQLite table people_embeddings(name PRIMARY KEY, embedding BLOB)
- Stores multiple embeddings per person for robust matching
- Handles people leaving/re-entering the frame
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
MODEL_PATH = "yolo11n.pt"
INPUT_FILE = "youtube_input23.mp4"
YOUTUBE_URL = "https://youtu.be/ulYDSTdbGa8?si=7KiFjKW3_A9HLKOv"
PEOPLE_DB = "people.db"
EMB_DB = "embeddings.db"
OUTPUT_FILE = "output_with_names_v10.mp4"
source = 0  # or "http://192.168.1.18:8080/video"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.82
UPDATE_ALPHA = 0.90
MIN_BOX_AREA = 1600
BATCH_WRITE_INTERVAL = 500

MAX_EMB_HISTORY = 10
TRACK_TIMEOUT = 300  # frames (~6 sec at 25fps)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FEATURE EXTRACTOR ----------------
print("🔍 Loading ResNet50 feature extractor...")
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_embedding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
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
        feats = feature_extractor(tensor)
        feats = feats.reshape(1, -1)
        feats = F.normalize(feats, p=2, dim=1)
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

def load_all_embeddings() -> Dict[str, List[np.ndarray]]:
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM people_embeddings")
    rows = c.fetchall()
    conn.close()
    memory: Dict[str, List[np.ndarray]] = {}
    for name, blob in rows:
        try:
            emb = blob_to_np(blob).astype(np.float32).flatten()
            memory[name] = [emb]
        except Exception:
            continue
    return memory

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
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def update_memory(name: str, emb: np.ndarray, memory: Dict[str, List[np.ndarray]]):
    if name not in memory:
        memory[name] = []
    memory[name].append(emb)
    if len(memory[name]) > MAX_EMB_HISTORY:
        memory[name].pop(0)

def robust_find_best_match(emb: np.ndarray, memory: Dict[str, List[np.ndarray]]) -> Tuple[Optional[str], float]:
    best_name = None
    best_sim = -1.0
    for name, embs_list in memory.items():
        for e in embs_list:
            sim = cosine_similarity(emb, e)
            if sim > best_sim:
                best_sim = sim
                best_name = name
    return best_name, best_sim

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
    memory = load_all_embeddings()  # name -> list of embeddings
    name_list = load_names_ordered()
    print(f"📚 Loaded {len(memory)} stored embeddings and {len(name_list)} names from people.db")

    assigned_names: Dict[int, str] = {}  # track_id -> name
    disappeared_tracks: Dict[int, Tuple[int, str]] = {}  # track_id -> (last_seen_frame, name)
    pending_updates: Dict[str, np.ndarray] = {}
    next_name_idx = 0
    used_names = set(memory.keys())

    while next_name_idx < len(name_list) and name_list[next_name_idx] in used_names:
        next_name_idx += 1

    video_path = download_youtube(YOUTUBE_URL, INPUT_FILE)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Unable to open video")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    yolo = YOLO(MODEL_PATH)
    yolo.to(device)

    frame_idx = 0
    print("🚀 Starting robust persistent person tracking (v3)")

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
                cls_name = yolo.names[int(cls_idx)]
                if cls_name != "person" or track_id is None:
                    continue
                track_id = int(track_id)

                emb = extract_embedding(frame, bbox)
                if emb is None:
                    continue

                # Try reconnecting disappeared tracks first
                reconnected = False
                for old_id, (last_seen, old_name) in list(disappeared_tracks.items()):
                    if frame_idx - last_seen <= TRACK_TIMEOUT:
                        sim = cosine_similarity(emb, memory[old_name][-1])
                        if sim >= MATCH_THRESHOLD:
                            name = old_name
                            assigned_names[track_id] = name
                            update_memory(name, emb, memory)
                            pending_updates[name] = emb
                            disappeared_tracks.pop(old_id)
                            print(f"🔁 Reconnected disappeared track {old_id} -> {track_id} as {name} (sim={sim:.3f})")
                            reconnected = True
                            break
                if reconnected:
                    continue

                # Check memory for best match
                match_name, sim = robust_find_best_match(emb, memory)
                if match_name is not None and sim >= MATCH_THRESHOLD:
                    name = match_name
                    assigned_names[track_id] = name
                    update_memory(name, emb, memory)
                    pending_updates[name] = emb
                    print(f"🔁 Reconnected track {track_id} -> {name} (sim={sim:.3f})")
                else:
                    # Assign new name
                    if next_name_idx < len(name_list):
                        name = name_list[next_name_idx]
                        next_name_idx += 1
                    else:
                        kne = 0
                        while True:
                            cand = f"Unknown_{kne}"
                            if cand not in used_names:
                                name = cand
                                break
                            kne += 1
                    assigned_names[track_id] = name
                    update_memory(name, emb, memory)
                    pending_updates[name] = emb
                    used_names.add(name)
                    print(f"🆕 Assigned new name {name} to track {track_id}")

                # Draw box + label
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, assigned_names.get(track_id, "..."),
                            (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            # Update disappeared_tracks
            current_ids = set(int(id) for id in ids if id is not None)
            for tid in list(assigned_names.keys()):
                if tid not in current_ids:
                    disappeared_tracks[tid] = (frame_idx, assigned_names[tid])
                    assigned_names.pop(tid)

            out.write(frame)
            preview = cv2.resize(frame, (640, 640)) if frame.shape[0] >= 640 and frame.shape[1] >= 640 else frame
            cv2.imshow("Persistent ReID v3", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_idx % BATCH_WRITE_INTERVAL == 0 and pending_updates:
                try:
                    bulk_update_embeddings({n: memory[n][-1] for n in pending_updates})
                    pending_updates.clear()
                    memory = load_all_embeddings()
                    print(f"💾 Flushed embeddings to DB at frame {frame_idx}")
                except Exception as e:
                    print("⚠️ Error writing embeddings to DB:", e)

    finally:
        if pending_updates:
            try:
                bulk_update_embeddings({n: memory[n][-1] for n in pending_updates})
                print("💾 Final embeddings flush complete.")
            except Exception as e:
                print("⚠️ Final DB flush error:", e)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Done. Output saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
