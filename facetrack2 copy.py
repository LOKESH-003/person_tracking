# file: yolo_person_persistent_tracking_v4.py
"""
Robust persistent multi-person tracking with body+face embeddings and ghost tracks:
- ResNet50 for body embedding
- face_recognition for face embedding
- SQLite DB for persistent embeddings
- Ghost tracks for disappeared people
- Guaranteed no new names on re-entry
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
import face_recognition
from typing import List, Tuple, Dict, Optional

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11n.pt"                   # YOLO weights
INPUT_FILE = "youtube_input23.mp4"          # local video path
YOUTUBE_URL = "https://youtu.be/ulYDSTdbGa8?si=7KiFjKW3_A9HLKOv"
PEOPLE_DB = "people.db"                     # ordered names
EMB_DB = "embeddings.db"                    # persistent embeddings
OUTPUT_FILE = "output_with_names_v7.mp4"
source = 0  # 0 for webcam or RTSP link

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.82      # cosine similarity threshold
UPDATE_ALPHA = 0.90         # exponential moving average
MIN_BOX_AREA = 1600
BATCH_WRITE_INTERVAL = 500
GHOST_TIMEOUT = 150          # frames to keep disappeared tracks alive (~6 sec at 25 FPS)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FEATURE EXTRACTORS ----------------
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

def extract_body_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(frame.shape[1]-1, x2)
    y1, y2 = max(0, y1), min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1)<MIN_BOX_AREA:
        return None
    crop = frame[y1:y2, x1:x2]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor)
        feats = feats.reshape(1, -1)
        feats = F.normalize(feats, p=2, dim=1)
        emb = feats.cpu().numpy().astype(np.float32).flatten()
    return emb

def extract_face_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    try:
        face_locations = face_recognition.face_locations(face_crop)
        if not face_locations:
            return None
        face_encodings = face_recognition.face_encodings(face_crop, known_face_locations=face_locations)
        if face_encodings:
            return face_encodings[0].astype(np.float32)
    except Exception:
        return None
    return None

def combine_embeddings(body_emb: Optional[np.ndarray], face_emb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if body_emb is None and face_emb is None:
        return None
    if body_emb is None:
        body_emb = np.zeros(2048, dtype=np.float32)
    if face_emb is None:
        face_emb = np.zeros(128, dtype=np.float32)
    combined = np.concatenate([body_emb, face_emb])
    return combined / (np.linalg.norm(combined)+1e-12)

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
            data.append((name, emb.astype(np.float32).flatten()))
        except:
            continue
    return data

def bulk_update_embeddings(updates: Dict[str, np.ndarray]):
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    rows = [(name, np_to_blob(emb)) for name, emb in updates.items()]
    c.executemany("INSERT OR REPLACE INTO people_embeddings(name, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

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

# ---------------- UTIL ----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size==0 or b.size==0:
        return -1.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def find_best_match(emb: np.ndarray, memory: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
    if emb is None or not memory:
        return None, -1.0
    names = [n for n,_ in memory]
    embs = np.stack([e for _,e in memory])
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-12)
    emb_norm = emb / (np.linalg.norm(emb)+1e-12)
    sims = np.dot(emb_norm, embs_norm.T)
    idx = int(np.argmax(sims))
    return names[idx], float(sims[idx])

# ---------------- VIDEO / YOLO SETUP ----------------
print(f"✅ Loading YOLO model to {device}...")
yolo = YOLO(MODEL_PATH)
yolo.to(device)

def download_youtube(url: str, save_as: str) -> str:
    if os.path.exists(save_as):
        print(f"🎬 Using cached: {save_as}")
        return save_as
    import yt_dlp
    print("📥 Downloading video...")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': save_as, 'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("✅ Downloaded.")
    return save_as

# ---------------- MAIN ----------------
def main():
    init_db()
    memory = load_all_embeddings()
    name_list = load_names_ordered()
    print(f"📚 Loaded {len(memory)} embeddings and {len(name_list)} names from people.db")

    assigned_names: Dict[int, str] = {}
    ghost_tracks: Dict[int, Dict] = {}  # track_id -> {emb, bbox, frames_missing}
    pending_updates: Dict[str, np.ndarray] = {}
    used_names = {n for n,_ in memory}
    next_name_idx = 0
    while next_name_idx < len(name_list) and name_list[next_name_idx] in used_names:
        next_name_idx += 1

    # load video
    video_path = download_youtube(YOUTUBE_URL, INPUT_FILE)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Unable to open video")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    frame_idx = 0

    print("🚀 Starting robust persistent person tracking with ghost tracks")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = yolo.track(frame, persist=True, verbose=False,
                                 conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            if results is None or len(results)==0 or results[0].boxes is None:
                out.write(frame)
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            if ids is None:
                ids = [None]*len(boxes)

            for bbox, cls_idx, track_id in zip(boxes, classes, ids):
                cls_name = yolo.names[int(cls_idx)]
                if cls_name != "person" or track_id is None:
                    continue
                track_id = int(track_id)

                body_emb = extract_body_embedding(frame, bbox)
                face_emb = extract_face_embedding(frame, bbox)
                emb = combine_embeddings(body_emb, face_emb)
                if emb is None:
                    continue

                # reconnect ghost tracks first
                matched = False
                for ghost_id, ghost in list(ghost_tracks.items()):
                    sim = cosine_similarity(emb, ghost['emb'])
                    if sim >= MATCH_THRESHOLD:
                        name = ghost['name']
                        assigned_names[track_id] = name
                        ghost_tracks.pop(ghost_id)
                        # update memory embedding
                        for i,(n,e) in enumerate(memory):
                            if n==name:
                                new_avg = UPDATE_ALPHA*e + (1-UPDATE_ALPHA)*emb
                                new_avg = new_avg / (np.linalg.norm(new_avg)+1e-12)
                                memory[i] = (n,new_avg.astype(np.float32))
                                pending_updates[n] = new_avg.astype(np.float32)
                                break
                        print(f"🔁 Reconnected disappeared track {track_id} -> {name} (sim={sim:.3f})")
                        matched = True
                        break

                if not matched:
                    if track_id in assigned_names:
                        name = assigned_names[track_id]
                        # update memory embedding
                        for i,(n,e) in enumerate(memory):
                            if n==name:
                                new_avg = UPDATE_ALPHA*e + (1-UPDATE_ALPHA)*emb
                                new_avg = new_avg / (np.linalg.norm(new_avg)+1e-12)
                                memory[i] = (n,new_avg.astype(np.float32))
                                pending_updates[n] = new_avg.astype(np.float32)
                                break
                    else:
                        # new person
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
                        memory.append((name, emb.astype(np.float32)))
                        pending_updates[name] = emb.astype(np.float32)
                        used_names.add(name)
                        print(f"🆕 Assigned new name {name} to track {track_id}")

                # draw box + label
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0),2)
                display_name = assigned_names.get(track_id, "...")
                cv2.putText(frame, display_name, (x1,max(15,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,0),2)

            # update ghost tracks
            current_ids = set([int(t) for t in ids if t is not None])
            for t_id in list(assigned_names.keys()):
                if t_id not in current_ids:
                    if t_id not in ghost_tracks:
                        ghost_tracks[t_id] = {
                            'emb': next((e for n,e in memory if n==assigned_names[t_id]), None),
                            'name': assigned_names[t_id],
                            'frames_missing': 1
                        }
                    else:
                        ghost_tracks[t_id]['frames_missing'] += 1
                    # remove after timeout
                    if ghost_tracks[t_id]['frames_missing'] > GHOST_TIMEOUT:
                        ghost_tracks.pop(t_id)
                        assigned_names.pop(t_id)

            out.write(frame)
            preview = cv2.resize(frame, (640,640)) if frame.shape[0]>=640 else frame
            cv2.imshow("Persistent ReID v4", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_idx % BATCH_WRITE_INTERVAL == 0 and pending_updates:
                try:
                    bulk_update_embeddings(pending_updates)
                    pending_updates.clear()
                    memory = load_all_embeddings()
                    print(f"💾 Flushed embeddings to DB at frame {frame_idx}")
                except Exception as e:
                    print("⚠️ Error writing embeddings to DB:", e)

    finally:
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


if __name__=="__main__":
    main()

