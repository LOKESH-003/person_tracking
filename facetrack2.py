# # file: yolo_person_persistent_tracking_v4.py
# """
# Robust persistent multi-person ReID with body + face embeddings (v4):
# - ResNet50 body embedding
# - Face embedding via face_recognition
# - Multi-embedding memory per person
# - Disappeared track recovery
# - DB persistence (SQLite)
# """

# import os
# import cv2
# import sqlite3
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image
# from ultralytics import YOLO
# import yt_dlp
# from typing import List, Tuple, Dict, Optional
# import face_recognition

# # ---------------- CONFIG ----------------
# MODEL_PATH = "yolo11n.pt"
# INPUT_FILE = "youtube_input23.mp4"
# YOUTUBE_URL = "https://youtu.be/ulYDSTdbGa8?si=7KiFjKW3_A9HLKOv"
# PEOPLE_DB = "people.db"
# EMB_DB = "embeddings.db"
# OUTPUT_FILE = "output_with_names_v6.mp4"
# source = 0  # 0 for webcam or IP camera URL

# CONF_THRESHOLD = 0.5
# IOU_THRESHOLD = 0.4
# MATCH_THRESHOLD = 0.82
# UPDATE_ALPHA = 0.90
# MIN_BOX_AREA = 1600
# BATCH_WRITE_INTERVAL = 500

# MAX_EMB_HISTORY = 10
# TRACK_TIMEOUT = 150  # frames (~6 sec at 25fps)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------------- FEATURE EXTRACTOR ----------------
# print("🔍 Loading ResNet50 feature extractor...")
# resnet = models.resnet50(pretrained=True)
# feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
# feature_extractor.eval().to(device)

# preprocess = transforms.Compose([
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # ---------------- EMBEDDING FUNCTIONS ----------------
# def extract_body_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
#     x1, y1, x2, y2 = map(int, bbox)
#     x1, x2 = max(0, x1), max(0, min(frame.shape[1]-1, x2))
#     y1, y2 = max(0, y1), max(0, min(frame.shape[0]-1, y2))
#     if x2 <= x1 or y2 <= y1:
#         return None
#     crop = frame[y1:y2, x1:x2]
#     if crop.size == 0 or (x2-x1)*(y2-y1) < MIN_BOX_AREA:
#         return None
#     img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#     tensor = preprocess(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feats = feature_extractor(tensor)
#         feats = feats.reshape(1, -1)
#         feats = F.normalize(feats, p=2, dim=1)
#         emb = feats.cpu().numpy().astype(np.float32).flatten()
#     return emb

# def extract_face_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
#     x1, y1, x2, y2 = map(int, bbox)
#     face_crop = frame[y1:y2, x1:x2]

#     # Convert to RGB
#     face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

#     # Detect face landmarks in cropped image
#     face_locations = face_recognition.face_locations(face_crop_rgb)

#     if not face_locations:
#         return None

#     # Only pass the cropped image; do NOT pass face_locations to avoid dlib errors
#     face_encodings = face_recognition.face_encodings(face_crop_rgb)

#     if not face_encodings:
#         return None

#     return face_encodings[0].astype(np.float32)


# def combine_embeddings(body_emb: np.ndarray, face_emb: Optional[np.ndarray]) -> np.ndarray:
#     if face_emb is None:
#         return body_emb
#     body_emb = body_emb / (np.linalg.norm(body_emb)+1e-12)
#     face_emb = face_emb / (np.linalg.norm(face_emb)+1e-12)
#     combined = np.concatenate([body_emb, face_emb])
#     return combined / (np.linalg.norm(combined)+1e-12)

# # ---------------- SQLITE HELPERS ----------------
# def init_db():
#     conn = sqlite3.connect(EMB_DB)
#     c = conn.cursor()
#     c.execute("""CREATE TABLE IF NOT EXISTS people_embeddings (
#                  name TEXT PRIMARY KEY,
#                  embedding BLOB
#                  )""")
#     conn.commit()
#     conn.close()

# def np_to_blob(arr: np.ndarray) -> bytes:
#     return arr.astype(np.float32).tobytes()

# def blob_to_np(blob: bytes) -> np.ndarray:
#     return np.frombuffer(blob, dtype=np.float32)

# def load_all_embeddings() -> Dict[str, List[np.ndarray]]:
#     conn = sqlite3.connect(EMB_DB)
#     c = conn.cursor()
#     c.execute("SELECT name, embedding FROM people_embeddings")
#     rows = c.fetchall()
#     conn.close()
#     memory: Dict[str,List[np.ndarray]] = {}
#     for name, blob in rows:
#         try:
#             emb = blob_to_np(blob).astype(np.float32).flatten()
#             memory[name] = [emb]
#         except Exception:
#             continue
#     return memory

# def bulk_update_embeddings(updates: Dict[str, np.ndarray]):
#     conn = sqlite3.connect(EMB_DB)
#     c = conn.cursor()
#     rows = [(name, np_to_blob(emb)) for name, emb in updates.items()]
#     c.executemany("INSERT OR REPLACE INTO people_embeddings(name, embedding) VALUES (?, ?)", rows)
#     conn.commit()
#     conn.close()

# # ---------------- UTIL ----------------
# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     if a is None or b is None or a.size==0 or b.size==0:
#         return -1.0
#     denom = np.linalg.norm(a)*np.linalg.norm(b)
#     if denom==0: return -1.0
#     return float(np.dot(a,b)/denom)

# def update_memory(name: str, emb: np.ndarray, memory: Dict[str,List[np.ndarray]]):
#     if name not in memory:
#         memory[name] = []
#     memory[name].append(emb)
#     if len(memory[name]) > MAX_EMB_HISTORY:
#         memory[name].pop(0)

# def robust_find_best_match(emb: np.ndarray, memory: Dict[str,List[np.ndarray]]) -> Tuple[Optional[str], float]:
#     best_name = None
#     best_sim = -1.0
#     for name, embs_list in memory.items():
#         for e in embs_list:
#             sim = cosine_similarity(emb, e)
#             if sim > best_sim:
#                 best_sim = sim
#                 best_name = name
#     return best_name, best_sim

# def download_youtube(url: str, save_as: str) -> str:
#     if os.path.exists(save_as):
#         print(f"🎬 Using cached: {save_as}")
#         return save_as
#     print("📥 Downloading video...")
#     ydl_opts = {'format': 'best[ext=mp4]/best','outtmpl':save_as,'quiet':True,'no_warnings':True}
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])
#     print("✅ Downloaded.")
#     return save_as

# def load_names_ordered() -> List[str]:
#     if not os.path.exists(PEOPLE_DB):
#         return []
#     conn = sqlite3.connect(PEOPLE_DB)
#     c = conn.cursor()
#     try:
#         c.execute("SELECT name FROM people ORDER BY id ASC")
#         rows = c.fetchall()
#         return [r[0] for r in rows]
#     finally:
#         conn.close()

# # ---------------- MAIN ----------------
# def main():
#     init_db()
#     memory = load_all_embeddings()
#     name_list = load_names_ordered()
#     print(f"📚 Loaded {len(memory)} embeddings, {len(name_list)} names from people.db")

#     assigned_names: Dict[int,str] = {}
#     disappeared_tracks: Dict[int,Tuple[int,str]] = {}
#     pending_updates: Dict[str,np.ndarray] = {}
#     next_name_idx = 0
#     used_names = set(memory.keys())
#     while next_name_idx < len(name_list) and name_list[next_name_idx] in used_names:
#         next_name_idx += 1

#     video_path = download_youtube(YOUTUBE_URL, INPUT_FILE)
#     cap = cv2.VideoCapture(source)
#     if not cap.isOpened():
#         print("❌ Unable to open video")
#         return
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

#     yolo = YOLO(MODEL_PATH)
#     yolo.to(device)

#     frame_idx = 0
#     print("🚀 Starting robust persistent person tracking with body+face embeddings")

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_idx += 1

#             results = yolo.track(frame, persist=True, verbose=False,
#                                  conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
#             if results is None or len(results)==0 or results[0].boxes is None:
#                 out.write(frame)
#                 continue

#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             ids = results[0].boxes.id
#             classes = results[0].boxes.cls.cpu().numpy().astype(int)
#             if ids is None:
#                 ids = [None]*len(boxes)

#             for bbox, cls_idx, track_id in zip(boxes, classes, ids):
#                 cls_name = yolo.names[int(cls_idx)]
#                 if cls_name != "person" or track_id is None:
#                     continue
#                 track_id = int(track_id)

#                 body_emb = extract_body_embedding(frame, bbox)
#                 face_emb = extract_face_embedding(frame, bbox)
#                 if body_emb is None:
#                     continue
#                 emb = combine_embeddings(body_emb, face_emb)

#                 # Reconnect disappeared tracks
#                 reconnected=False
#                 for old_id, (last_seen, old_name) in list(disappeared_tracks.items()):
#                     if frame_idx - last_seen <= TRACK_TIMEOUT:
#                         sim = cosine_similarity(emb, memory[old_name][-1])
#                         if sim >= MATCH_THRESHOLD:
#                             name=old_name
#                             assigned_names[track_id]=name
#                             update_memory(name, emb, memory)
#                             pending_updates[name]=emb
#                             disappeared_tracks.pop(old_id)
#                             print(f"🔁 Reconnected disappeared track {old_id} -> {track_id} as {name} (sim={sim:.3f})")
#                             reconnected=True
#                             break
#                 if reconnected:
#                     continue

#                 # Check memory
#                 match_name, sim = robust_find_best_match(emb, memory)
#                 if match_name is not None and sim >= MATCH_THRESHOLD:
#                     name = match_name
#                     assigned_names[track_id] = name
#                     update_memory(name, emb, memory)
#                     pending_updates[name] = emb
#                 else:
#                     # Assign new name
#                     if next_name_idx < len(name_list):
#                         name = name_list[next_name_idx]
#                         next_name_idx += 1
#                     else:
#                         kne = 0
#                         while True:
#                             cand = f"Unknown_{kne}"
#                             if cand not in used_names:
#                                 name = cand
#                                 break
#                             kne += 1
#                     assigned_names[track_id]=name
#                     used_names.add(name)
#                     update_memory(name, emb, memory)
#                     pending_updates[name]=emb
#                     print(f"🆕 Assigned new name {name} to track {track_id}")

#                 # draw box + label
#                 x1,y1,x2,y2 = map(int,bbox)
#                 cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
#                 display_name = assigned_names.get(track_id,"...")
#                 cv2.putText(frame, display_name, (x1,max(15,y1-10)),
#                             cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,0),2)

#             # Update disappeared tracks
#             current_ids = set(ids)
#             for tid in list(assigned_names.keys()):
#                 if tid not in current_ids:
#                     disappeared_tracks[tid]=(frame_idx, assigned_names[tid])
#                     assigned_names.pop(tid)

#             out.write(frame)
#             preview = cv2.resize(frame,(640,640)) if frame.shape[0]>=640 and frame.shape[1]>=640 else frame
#             cv2.imshow("Persistent ReID v4", preview)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             if frame_idx % BATCH_WRITE_INTERVAL == 0 and pending_updates:
#                 try:
#                     bulk_update_embeddings(pending_updates)
#                     pending_updates.clear()
#                     memory = load_all_embeddings()
#                     print(f"💾 Flushed embeddings at frame {frame_idx}")
#                 except Exception as e:
#                     print("⚠️ Error writing DB:", e)

#     finally:
#         if pending_updates:
#             try:
#                 bulk_update_embeddings(pending_updates)
#                 print("💾 Final embeddings flush complete.")
#             except Exception as e:
#                 print("⚠️ Final DB flush error:", e)
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         print("✅ Done. Output saved to:", OUTPUT_FILE)

# if __name__ == "__main__":
#     main()

# file: yolo_person_persistent_tracking_v4.py
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
import face_recognition
from typing import List, Tuple, Dict, Optional

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11n.pt"                  # YOLO detector weights
INPUT_FILE = "youtube_input23.mp4"         # local video path
YOUTUBE_URL = "https://youtu.be/ulYDSTdbGa8?si=7KiFjKW3_A9HLKOv"
PEOPLE_DB = "people.db"                    # ordered list of names
EMB_DB = "embeddings.db"                   # persistent embeddings DB
OUTPUT_FILE = "output_with_names_v6.mp4"
source = 0
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.82
UPDATE_ALPHA = 0.90
MIN_BOX_AREA = 1600
BATCH_WRITE_INTERVAL = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FEATURE EXTRACTOR ----------------
print("🔍 Loading ResNet50 feature extractor...")
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def extract_body_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    x1,y1,x2,y2 = map(int,bbox)
    x1,x2 = max(0,x1), max(0,min(frame.shape[1]-1,x2))
    y1,y2 = max(0,y1), max(0,min(frame.shape[0]-1,y2))
    if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)<MIN_BOX_AREA:
        return None
    crop = frame[y1:y2,x1:x2]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor).reshape(1,-1)
        feats = F.normalize(feats,p=2,dim=1)
        return feats.cpu().numpy().astype(np.float32).flatten()

def extract_face_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    x1,y1,x2,y2 = map(int,bbox)
    face_crop = frame[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(face_rgb)
    if not face_locations:
        return None
    encodings = face_recognition.face_encodings(face_rgb)
    if not encodings:
        return None
    return encodings[0].astype(np.float32)

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
            emb = blob_to_np(blob).astype(np.float32).flatten()
            data.append((name, emb))
        except Exception:
            continue
    return data

def bulk_update_embeddings(updates: Dict[str,np.ndarray]):
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    rows = [(name, np_to_blob(emb)) for name,emb in updates.items()]
    c.executemany("INSERT OR REPLACE INTO people_embeddings(name,embedding) VALUES (?,?)", rows)
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
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    if denom==0: return -1.0
    return float(np.dot(a,b)/denom)

def find_best_match(emb: np.ndarray, memory: List[Tuple[str,np.ndarray]]) -> Tuple[Optional[str],float]:
    if emb is None or not memory: return None,-1.0
    names = [n for n,_ in memory]
    embs = np.stack([e for _,e in memory])
    embs_norm = embs/ (np.linalg.norm(embs,axis=1,keepdims=True)+1e-12)
    emb_norm = emb/ (np.linalg.norm(emb)+1e-12)
    sims = np.dot(emb_norm, embs_norm.T)
    idx = int(np.argmax(sims))
    return names[idx], float(sims[idx])

def combine_embeddings(body_emb: Optional[np.ndarray], face_emb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if body_emb is None and face_emb is None:
        return None
    if body_emb is None:
        body_emb = np.zeros(2048, dtype=np.float32)
    if face_emb is None:
        face_emb = np.zeros(128, dtype=np.float32)
    combined = np.concatenate([body_emb, face_emb])
    return combined / (np.linalg.norm(combined)+1e-12)


# ---------------- VIDEO / YOLO SETUP ----------------
print(f"✅ Loading YOLO model to {device}...")
yolo = YOLO(MODEL_PATH)
yolo.to(device)

def download_youtube(url: str, save_as: str) -> str:
    if os.path.exists(save_as):
        print(f"🎬 Using cached: {save_as}")
        return save_as
    print("📥 Downloading video...")
    ydl_opts = {'format':'best[ext=mp4]/best','outtmpl':save_as,'quiet':True,'no_warnings':True}
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
    assigned_names: Dict[int,str] = {}
    pending_updates: Dict[str,np.ndarray] = {}
    last_bboxes: Dict[int,Tuple[int,int,int,int]] = {}
    disappeared_tracks: Dict[int,Tuple[int,str]] = {}
    next_name_idx = 0
    used_names = {n for n,_ in memory}
    while next_name_idx<len(name_list) and name_list[next_name_idx] in used_names:
        next_name_idx+=1

    video_path = download_youtube(YOUTUBE_URL, INPUT_FILE)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Unable to open video"); return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    frame_idx = 0
    print("🚀 Starting robust persistent person tracking with body+face embeddings")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx+=1

            results = yolo.track(frame, persist=True, verbose=False,
                                 conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            if results is None or len(results)==0 or results[0].boxes is None:
                out.write(frame); continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            if ids is None: ids = [None]*len(boxes)

            for bbox, cls_idx, track_id in zip(boxes, classes, ids):
                cls_name = yolo.names[int(cls_idx)]
                if cls_name!="person" or track_id is None: continue
                track_id = int(track_id)
                last_bboxes[track_id] = bbox

                body_emb = extract_body_embedding(frame, bbox)
                face_emb = extract_face_embedding(frame, bbox)
                emb = combine_embeddings(body_emb, face_emb)
                if emb is None: continue

                if track_id in assigned_names:
                    name = assigned_names[track_id]
                    # update memory
                    for i,(n,e) in enumerate(memory):
                        if n==name:
                            new_avg = UPDATE_ALPHA*e + (1-UPDATE_ALPHA)*emb
                            memory[i] = (n,new_avg/ (np.linalg.norm(new_avg)+1e-12).astype(np.float32))
                            pending_updates[n] = memory[i][1]
                            break
                else:
                    match_name, sim = find_best_match(emb, memory)
                    if match_name is not None and sim>=MATCH_THRESHOLD:
                        name = match_name
                        # update memory
                        for i,(n,e) in enumerate(memory):
                            if n==name:
                                new_avg = UPDATE_ALPHA*e + (1-UPDATE_ALPHA)*emb
                                memory[i] = (n,new_avg/ (np.linalg.norm(new_avg)+1e-12).astype(np.float32))
                                pending_updates[n] = memory[i][1]
                                break
                        assigned_names[track_id]=name
                        print(f"🔁 Reconnected disappeared track {track_id} -> {name} (sim={sim:.3f})")
                    else:
                        # new person
                        if next_name_idx<len(name_list):
                            name = name_list[next_name_idx]
                            next_name_idx+=1
                        else:
                            kne=0
                            while True:
                                cand=f"Unknown_{kne}"
                                if cand not in used_names: name=cand; break
                                kne+=1
                        assigned_names[track_id]=name
                        used_names.add(name)
                        memory.append((name, emb/ (np.linalg.norm(emb)+1e-12)))
                        pending_updates[name]=memory[-1][1]
                        print(f"🆕 Assigned new name {name} to track {track_id}")

                # draw box + label
                x1,y1,x2,y2 = map(int,bbox)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
                cv2.putText(frame,name,(x1,max(15,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,0),2)

            # ghost boxes for disappeared tracks
            for tid,name in assigned_names.items():
                if tid not in ids and tid in last_bboxes:
                    x1,y1,x2,y2 = map(int,last_bboxes[tid])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,120,0),1)
                    cv2.putText(frame,name,(x1,max(15,y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,150,0),1)

            out.write(frame)
            preview = cv2.resize(frame,(640,640)) if frame.shape[0]>=640 and frame.shape[1]>=640 else frame
            cv2.imshow("Persistent ReID v4", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            if frame_idx % BATCH_WRITE_INTERVAL==0 and pending_updates:
                try:
                    bulk_update_embeddings(pending_updates)
                    pending_updates.clear()
                    memory = load_all_embeddings()
                    print(f"💾 Flushed embeddings at frame {frame_idx}")
                except Exception as e:
                    print("⚠️ Error writing DB:",e)

    finally:
        if pending_updates:
            try:
                bulk_update_embeddings(pending_updates)
                print("💾 Final embeddings flush complete.")
            except Exception as e:
                print("⚠️ Final DB flush error:", e)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Done. Output saved to:", OUTPUT_FILE)

if __name__=="__main__":
    main()
