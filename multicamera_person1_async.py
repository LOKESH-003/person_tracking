# # file: multicam_reid_fixed.py
# from typing import List, Tuple, Dict, Optional
# import os, threading, sqlite3, time, cv2, numpy as np, torch, torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image
# from ultralytics import YOLO
# import face_recognition
# from concurrent.futures import ThreadPoolExecutor

# YOLO_WEIGHTS = "yolo11n.pt"
# GALLERY_DIR = "gallery"
# CAMERA_SOURCES = ["http://192.168.1.24:8080/video", 0]
# EMB_DB = "embeddings.db"
# FACE_DB = "faces.db"
# OUTPUT_DIR = "outputs"
# CONF_THRESHOLD = 0.5
# IOU_THRESHOLD = 0.4
# MATCH_THRESHOLD = 0.82
# FACE_MATCH_THRESHOLD = 0.50
# UPDATE_ALPHA = 0.90
# MIN_BOX_AREA = 1600
# device = "cuda" if torch.cuda.is_available() else "cpu"

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

# def extract_body_embedding(frame, bbox):
#     x1, y1, x2, y2 = map(int, bbox)
#     if (x2-x1)*(y2-y1) < MIN_BOX_AREA:
#         return None
#     crop = frame[y1:y2, x1:x2]
#     if crop.size == 0:
#         return None
#     img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#     tensor = preprocess(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feats = feature_extractor(tensor).reshape(1, -1)
#         feats = F.normalize(feats, p=2, dim=1)
#     return feats.cpu().numpy().astype(np.float32).flatten()

# def extract_face_embedding_from_crop(crop_rgb):
#     boxes = face_recognition.face_locations(crop_rgb, model="hog")
#     if not boxes:
#         return None
#     encs = face_recognition.face_encodings(crop_rgb, boxes)
#     if not encs:
#         return None
#     return np.array(encs[0], dtype=np.float32)

# def init_dbs():
#     for db_name, table in [(EMB_DB, "people_embeddings"), (FACE_DB, "face_embeddings")]:
#         conn = sqlite3.connect(db_name)
#         c = conn.cursor()
#         c.execute(f"CREATE TABLE IF NOT EXISTS {table} (name TEXT PRIMARY KEY, embedding BLOB)")
#         conn.commit()
#         conn.close()

# def np_to_blob(arr): return arr.astype(np.float32).tobytes()
# def blob_to_np(blob): return np.frombuffer(blob, dtype=np.float32)

# def load_all_embeddings(db, table):
#     conn = sqlite3.connect(db)
#     c = conn.cursor()
#     c.execute(f"SELECT name, embedding FROM {table}")
#     data = [(n, blob_to_np(b).astype(np.float32)) for n,b in c.fetchall()]
#     conn.close()
#     return data

# def bulk_update(db, table, updates):
#     if not updates: return
#     conn = sqlite3.connect(db)
#     c = conn.cursor()
#     c.executemany(f"INSERT OR REPLACE INTO {table}(name, embedding) VALUES (?, ?)",
#                   [(n, np_to_blob(e)) for n,e in updates.items()])
#     conn.commit(); conn.close()

# def build_face_gallery(gallery_dir):
#     if not os.path.exists(gallery_dir):
#         print("⚠️ Gallery not found:", gallery_dir)
#         return
#     face_updates = {}
#     for person in os.listdir(gallery_dir):
#         pdir = os.path.join(gallery_dir, person)
#         if not os.path.isdir(pdir): continue
#         encs=[]
#         for fn in os.listdir(pdir):
#             if not fn.lower().endswith((".jpg",".jpeg",".png")): continue
#             img = face_recognition.load_image_file(os.path.join(pdir, fn))
#             boxes = face_recognition.face_locations(img, model="hog")
#             es = face_recognition.face_encodings(img, boxes)
#             if es: encs.append(es[0])
#         if encs:
#             avg = np.mean(np.stack(encs), axis=0)
#             avg /= (np.linalg.norm(avg)+1e-12)
#             face_updates[person] = avg.astype(np.float32)
#             print(f"📸 Built face embedding for {person} ({len(encs)} imgs)")
#     bulk_update(FACE_DB, "face_embeddings", face_updates)

# def cosine_similarity(a, b):
#     denom = np.linalg.norm(a)*np.linalg.norm(b)
#     if denom==0: return -1
#     return float(np.dot(a,b)/denom)

# def find_best_body_match(emb, memory):
#     if emb is None or not memory: return None, -1
#     embs = np.stack([e for _,e in memory])
#     embs /= (np.linalg.norm(embs,axis=1,keepdims=True)+1e-12)
#     emb /= (np.linalg.norm(emb)+1e-12)
#     sims = np.dot(emb, embs.T)
#     idx=int(np.argmax(sims))
#     return memory[idx][0], float(sims[idx])

# def find_best_face_match(face_emb, memory):
#     if face_emb is None or not memory: return None, 1e6
#     mem = np.stack([e for _,e in memory])
#     dists = np.linalg.norm(mem-face_emb,axis=1)
#     idx=int(np.argmin(dists))
#     return memory[idx][0], float(dists[idx])

# class MultiCamReID:
#     def __init__(self, sources: List):
#         print(f"✅ Loading YOLO model to {device}...")
#         self.yolo = YOLO(YOLO_WEIGHTS)
#         self.yolo.to(device)
#         init_dbs()
#         self.sources = sources
#         self.lock = threading.Lock()
#         self.body_memory = load_all_embeddings(EMB_DB,"people_embeddings")
#         self.face_memory = load_all_embeddings(FACE_DB,"face_embeddings")
#         self.frames: Dict[int, np.ndarray] = {}
#         self.stop_flag=False
#         self.executor = ThreadPoolExecutor(max_workers=len(sources))
#         os.makedirs(OUTPUT_DIR, exist_ok=True)

#     def process_camera(self, cam_idx, source):
#         try:
#             cap = cv2.VideoCapture(source)
#             if not cap.isOpened():
#                 print(f"❌ Camera {cam_idx} failed to open")
#                 return
#             fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#             w,h=int(cap.get(3)),int(cap.get(4))
#             out = cv2.VideoWriter(os.path.join(OUTPUT_DIR,f"cam{cam_idx}.mp4"),
#                                   cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
#             print(f"🎥 Camera[{cam_idx}] started")
#             while not self.stop_flag:
#                 ret, frame = cap.read()
#                 if not ret: break
#                 results = self.yolo.track(frame, persist=True, verbose=False,
#                                           conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
#                 if results and len(results)>0 and results[0].boxes is not None:
#                     boxes = results[0].boxes.xyxy.cpu().numpy()
#                     ids = results[0].boxes.id
#                     classes = results[0].boxes.cls.cpu().numpy().astype(int)
#                     ids = ids if ids is not None else [None]*len(boxes)
#                     for bbox, cls_idx, track_id in zip(boxes, classes, ids):
#                         if self.yolo.names[int(cls_idx)]!="person" or track_id is None:
#                             continue
#                         x1,y1,x2,y2=map(int,bbox)
#                         crop=frame[y1:y2,x1:x2]
#                         face_emb=extract_face_embedding_from_crop(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
#                         name=None
#                         if face_emb is not None and self.face_memory:
#                             fname,dist=find_best_face_match(face_emb,self.face_memory)
#                             if dist<=FACE_MATCH_THRESHOLD: name=fname
#                         if name is None:
#                             emb=extract_body_embedding(frame,bbox)
#                             if emb is not None and self.body_memory:
#                                 bname,sim=find_best_body_match(emb,self.body_memory)
#                                 if sim>=MATCH_THRESHOLD: name=bname
#                         label=name or f"Unknown_{cam_idx}_{int(track_id)}"
#                         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
#                         cv2.putText(frame,label,(x1,max(15,y1-10)),
#                                     cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,0),2)
#                 out.write(frame)
#                 with self.lock:
#                     self.frames[cam_idx]=cv2.resize(frame,(640,360))
#             cap.release(); out.release()
#         except Exception as e:
#             print(f"⚠️ Camera[{cam_idx}] crashed:", e)

#     def show_all(self):
#         while not self.stop_flag:
#             with self.lock:
#                 frames=[self.frames[i] for i in sorted(self.frames.keys())]
#             if frames:
#                 n=len(frames)
#                 cols=int(np.ceil(np.sqrt(n)))
#                 rows=int(np.ceil(n/cols))
#                 h,w=frames[0].shape[:2]
#                 grid=np.zeros((rows*h,cols*w,3),dtype=np.uint8)
#                 for idx,f in enumerate(frames):
#                     r,c=divmod(idx,cols)
#                     grid[r*h:(r+1)*h, c*w:(c+1)*w]=f
#                 cv2.imshow("All Cameras",grid)
#             if cv2.waitKey(1)&0xFF==ord('q'):
#                 self.stop_flag=True
#                 break
#             time.sleep(0.03)
#         cv2.destroyAllWindows()

#     def start(self):
#         for i,src in enumerate(self.sources):
#             self.executor.submit(self.process_camera,i,src)
#         self.show_all()
#         self.stop_flag=True
#         print("🛑 Stopping all cameras...")
#         time.sleep(1)

# def main():
#     init_dbs()
#     build_face_gallery(GALLERY_DIR)
#     mgr=MultiCamReID(CAMERA_SOURCES)
#     mgr.start()
#     print("✅ Done. Outputs saved in",OUTPUT_DIR)

# if __name__=="__main__":
#     main()


from typing import List, Tuple, Dict, Optional
import os
import threading
import sqlite3
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import face_recognition
from concurrent.futures import ThreadPoolExecutor

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "yolo11n.pt"
GALLERY_DIR = "gallery"
CAMERA_SOURCES = ["http://192.168.1.24:8080/video", 0]   # 👈 Add your cameras here
EMB_DB = "embeddings.db"
FACE_DB = "faces.db"
OUTPUT_DIR = "outputs"

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.82
FACE_MATCH_THRESHOLD = 0.50
UPDATE_ALPHA = 0.90
MIN_BOX_AREA = 400
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

def extract_body_embedding(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(frame.shape[1]-1, x2)
    y1, y2 = max(0, y1), min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2-x1)*(y2-y1) < MIN_BOX_AREA:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor)
        feats = feats.reshape(1, -1)
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32).flatten()

def extract_face_embedding_from_crop(crop_rgb: np.ndarray) -> Optional[np.ndarray]:
    boxes = face_recognition.face_locations(crop_rgb, model="hog")
    if not boxes:
        return None
    encs = face_recognition.face_encodings(crop_rgb, boxes)
    if not encs:
        return None
    return np.array(encs[0], dtype=np.float32)

# ---------------- SQLITE HELPERS ----------------
def init_dbs():
    for db_name, table, field in [
        (EMB_DB, "people_embeddings", "embedding"),
        (FACE_DB, "face_embeddings", "embedding")
    ]:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                      name TEXT PRIMARY KEY,
                      {field} BLOB)""")
        conn.commit()
        conn.close()

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def load_all_embeddings(db, table) -> List[Tuple[str, np.ndarray]]:
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute(f"SELECT name, embedding FROM {table}")
    rows = c.fetchall()
    conn.close()
    out = []
    for name, blob in rows:
        try:
            emb = blob_to_np(blob).astype(np.float32).flatten()
            out.append((name, emb))
        except Exception:
            continue
    return out

def bulk_update(db, table, updates: Dict[str, np.ndarray]):
    if not updates:
        return
    conn = sqlite3.connect(db)
    c = conn.cursor()
    rows = [(n, np_to_blob(e)) for n, e in updates.items()]
    c.executemany(f"INSERT OR REPLACE INTO {table}(name, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

# ---------------- GALLERY ----------------
def build_face_gallery(gallery_dir: str) -> Dict[str, np.ndarray]:
    face_updates = {}
    if not os.path.exists(gallery_dir):
        print("⚠️ Gallery not found:", gallery_dir)
        return face_updates
    for person in os.listdir(gallery_dir):
        pdir = os.path.join(gallery_dir, person)
        if not os.path.isdir(pdir):
            continue
        encs = []
        for fn in os.listdir(pdir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(pdir, fn)
            img = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(img, model="hog")
            if not boxes:
                continue
            es = face_recognition.face_encodings(img, boxes)
            if es:
                encs.append(es[0])
        if encs:
            avg = np.mean(np.stack(encs), axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-12)
            face_updates[person] = avg.astype(np.float32)
            print(f"📸 Built face embedding for {person} ({len(encs)} images)")
    if face_updates:
        bulk_update(FACE_DB, "face_embeddings", face_updates)
    return face_updates

# ---------------- MATCHING ----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def find_best_body_match(emb, memory):
    if emb is None or not memory:
        return None, -1.0
    embs = np.stack([e for _, e in memory])
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-12)
    emb_norm = emb / (np.linalg.norm(emb)+1e-12)
    sims = np.dot(emb_norm, embs_norm.T)
    idx = int(np.argmax(sims))
    return memory[idx][0], float(sims[idx])

def find_best_face_match(face_emb, face_memory):
    if face_emb is None or not face_memory:
        return None, 1e6
    mem = np.stack([e for _, e in face_memory])
    dists = np.linalg.norm(mem - face_emb.reshape(1,-1), axis=1)
    idx = int(np.argmin(dists))
    return face_memory[idx][0], float(dists[idx])

# ---------------- MULTI-CAMERA ----------------
class MultiCamReID:
    def __init__(self, sources: List):
        print(f"✅ Loading YOLO model ({YOLO_WEIGHTS}) on {device}...")
        self.yolo = YOLO(YOLO_WEIGHTS)
        self.yolo.to(device)
        init_dbs()
        self.sources = sources
        self.lock = threading.Lock()
        self.body_memory = load_all_embeddings(EMB_DB, "people_embeddings")
        self.face_memory = load_all_embeddings(FACE_DB, "face_embeddings")
        self.frames: Dict[int, np.ndarray] = {}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=len(sources))
        self.stop_flag = False

    def process_camera(self, cam_idx, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Camera {cam_idx} failed to open")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f"cam{cam_idx}.mp4"),
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        print(f"🎥 Camera[{cam_idx}] started")

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # --- PERSON DETECTION ONLY ---
            results = self.yolo.track(frame, persist=True, verbose=False,
                                      conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                if ids is None:
                    ids = [None]*len(boxes)

                for bbox, cls_idx, track_id in zip(boxes, classes, ids):
                    cls_name = self.yolo.names[int(cls_idx)]
                    if cls_name.lower() != "person" or track_id is None:
                        continue

                    x1,y1,x2,y2 = map(int,bbox)
                    crop = frame[y1:y2, x1:x2]

                    # try face first
                    face_emb = extract_face_embedding_from_crop(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    name = None
                    if face_emb is not None and self.face_memory:
                        fname, dist = find_best_face_match(face_emb, self.face_memory)
                        if dist <= FACE_MATCH_THRESHOLD:
                            name = fname
                    if name is None:
                        emb = extract_body_embedding(frame, bbox)
                        if emb is not None and self.body_memory:
                            bname, sim = find_best_body_match(emb, self.body_memory)
                            if sim >= MATCH_THRESHOLD:
                                name = bname
                    label = name or f"Person_{cam_idx}_{int(track_id)}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)
                    cv2.putText(frame, label, (x1, max(15,y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            out.write(frame)
            with self.lock:
                self.frames[cam_idx] = cv2.resize(frame, (640,360))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag = True
                break

        cap.release()
        out.release()

    def show_all(self):
        """Combine frames from all cameras"""
        while not self.stop_flag:
            with self.lock:
                frames = [self.frames[i] for i in sorted(self.frames.keys()) if i in self.frames]
            if frames:
                n = len(frames)
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                h, w = frames[0].shape[:2]
                grid = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
                for idx, f in enumerate(frames):
                    r, c = divmod(idx, cols)
                    grid[r*h:(r+1)*h, c*w:(c+1)*w] = f
                cv2.imshow("All Cameras", grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag = True
                break
            time.sleep(0.03)
        cv2.destroyAllWindows()

    def start(self):
        for i, src in enumerate(self.sources):
            self.executor.submit(self.process_camera, i, src)
        self.show_all()
        self.stop_flag = True
        print("🛑 Stopping all cameras...")
        time.sleep(1)

# ---------------- MAIN ----------------
def main():
    init_dbs()
    build_face_gallery(GALLERY_DIR)
    mgr = MultiCamReID(CAMERA_SOURCES)
    mgr.start()
    print("✅ Done. Outputs saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
