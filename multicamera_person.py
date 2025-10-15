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
import yt_dlp
from concurrent.futures import ThreadPoolExecutor

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "yolo11n.pt"
GALLERY_DIR = "gallery"            # subfolders per person with images (multi-angle)
CAMERA_SOURCES = ["http://192.168.1.24:8080/video",0]               # list of camera sources (0,1 or rtsp/http urls) e.g. [0, "rtsp://..."]
INPUT_FILE = "youtube_input23.mp4" # if you want to download a YouTube video for testing
YOUTUBE_URL = "https://youtu.be/ulYDSTdbGa8?si=7KiFjKW3_A9HLKOv"
EMB_DB = "embeddings.db"           # body embeddings
FACE_DB = "faces.db"               # face embeddings
PEOPLE_DB = "people.db"            # ordered list of names (as you had)
OUTPUT_DIR = "outputs"             # per-camera output files stored here
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.82             # cosine for body embeddings
FACE_MATCH_THRESHOLD = 0.50        # face distance threshold (face_recognition: typical ~0.6)
UPDATE_ALPHA = 0.90
MIN_BOX_AREA = 1600
BATCH_WRITE_INTERVAL = 500         # total frames processed across all cams before flush
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
    x1,y1,x2,y2 = map(int, bbox)
    x1, x2 = max(0, x1), max(0, min(frame.shape[1]-1, x2))
    y1, y2 = max(0, y1), max(0, min(frame.shape[0]-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2-x1) * (y2-y1) < MIN_BOX_AREA:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor)      # [1,2048,1,1]
        feats = feats.reshape(1, -1)
        feats = F.normalize(feats, p=2, dim=1)
        emb = feats.cpu().numpy().astype(np.float32).flatten()
    return emb

def extract_face_embedding_from_crop(crop_rgb: np.ndarray) -> Optional[np.ndarray]:
    # face_recognition expects RGB arrays
    boxes = face_recognition.face_locations(crop_rgb, model="hog")
    if not boxes:
        return None
    encs = face_recognition.face_encodings(crop_rgb, boxes)
    if not encs:
        return None
    # if multiple faces in crop, return the largest (first)
    return np.array(encs[0], dtype=np.float32)

# ---------------- SQLITE HELPERS ----------------
def init_dbs():
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS people_embeddings (
                 name TEXT PRIMARY KEY,
                 embedding BLOB
                 )""")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(FACE_DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS face_embeddings (
                 name TEXT PRIMARY KEY,
                 embedding BLOB
                 )""")
    conn.commit()
    conn.close()

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def load_all_body_embeddings() -> List[Tuple[str, np.ndarray]]:
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM people_embeddings")
    rows = c.fetchall()
    conn.close()
    out=[]
    for name, blob in rows:
        try:
            emb=blob_to_np(blob).astype(np.float32).flatten()
            out.append((name, emb))
        except Exception:
            continue
    return out

def load_all_face_embeddings() -> List[Tuple[str, np.ndarray]]:
    conn = sqlite3.connect(FACE_DB)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM face_embeddings")
    rows = c.fetchall()
    conn.close()
    out=[]
    for name, blob in rows:
        try:
            emb=blob_to_np(blob).astype(np.float32).flatten()
            out.append((name, emb))
        except Exception:
            continue
    return out

def bulk_update_body(updates: Dict[str, np.ndarray]):
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    rows = [(n, np_to_blob(e)) for n,e in updates.items()]
    c.executemany("INSERT OR REPLACE INTO people_embeddings(name, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

def bulk_update_faces(updates: Dict[str, np.ndarray]):
    conn = sqlite3.connect(FACE_DB)
    c = conn.cursor()
    rows = [(n, np_to_blob(e)) for n,e in updates.items()]
    c.executemany("INSERT OR REPLACE INTO face_embeddings(name, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

# ---------------- GALLERY -> FACE DB ----------------
def build_face_gallery(gallery_dir: str) -> Dict[str, np.ndarray]:
    """
    Walk gallery_dir: each subfolder is a person's name; embed all images and average.
    Returns dict name->face_embedding (normalized).
    """
    face_db_updates={}
    if not os.path.exists(gallery_dir):
        print("⚠️ Gallery directory not found:", gallery_dir)
        return face_db_updates
    for person in os.listdir(gallery_dir):
        pdir = os.path.join(gallery_dir, person)
        if not os.path.isdir(pdir):
            continue
        encs=[]
        for fn in os.listdir(pdir):
            if not fn.lower().endswith((".jpg",".png",".jpeg")):
                continue
            path = os.path.join(pdir, fn)
            img = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(img, model="hog")
            if not boxes:
                continue
            es = face_recognition.face_encodings(img, boxes)
            if not es:
                continue
            encs.append(es[0])
        if encs:
            avg = np.mean(np.stack(encs), axis=0)
            # normalize L2
            avg = avg / (np.linalg.norm(avg) + 1e-12)
            face_db_updates[person] = avg.astype(np.float32)
            print(f"📸 Gallery: built face embedding for {person} from {len(encs)} images")
    # persist to face DB
    if face_db_updates:
        bulk_update_faces(face_db_updates)
    return face_db_updates

# ---------------- MATCHING UTIL ----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size==0 or b.size==0:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def find_best_body_match(emb: np.ndarray, memory: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
    if emb is None or len(memory)==0:
        return None, -1.0
    embs = np.stack([e for _,e in memory])
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-12)
    emb_norm = emb / (np.linalg.norm(emb)+1e-12)
    sims = np.dot(emb_norm, embs_norm.T)
    idx = int(np.argmax(sims))
    return memory[idx][0], float(sims[idx])

def find_best_face_match(face_emb: np.ndarray, face_memory: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
    # face_recognition uses euclidean distance; convert to similarity by -distance
    if face_emb is None or len(face_memory)==0:
        return None, 1e6
    mem = np.stack([e for _,e in face_memory])  # shape [N, 128]
    # compute euclidean distances
    dists = np.linalg.norm(mem - face_emb.reshape(1,-1), axis=1)
    idx = int(np.argmin(dists))
    return face_memory[idx][0], float(dists[idx])

# ---------------- MULTI-CAMERA PROCESSING ----------------
class MultiCamReID:
    def __init__(self, camera_sources: List, yolo_weights: str):
        self.camera_sources = camera_sources
        print(f"✅ Loading YOLO model ({yolo_weights}) to {device}...")
        self.yolo = YOLO(yolo_weights)
        self.yolo.to(device)
        self.lock = threading.Lock()
        init_dbs()
        # load memories (shared)
        self.body_memory = load_all_body_embeddings()      # list[(name, emb)]
        self.face_memory = load_all_face_embeddings()      # list[(name, emb)]
        self.pending_body_updates: Dict[str, np.ndarray] = {}
        self.pending_face_updates: Dict[str, np.ndarray] = {}
        self.used_names = {n for n,_ in self.body_memory}
        self.assigned_names: Dict[Tuple[int,int], str] = {}  # (cam_idx, track_id) -> name
        self.name_list = self._load_names_ordered()
        self.next_name_idx = 0
        while self.next_name_idx < len(self.name_list) and self.name_list[self.next_name_idx] in self.used_names:
            self.next_name_idx += 1
        self.frame_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=len(camera_sources))
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _load_names_ordered(self) -> List[str]:
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

    def process_camera(self, cam_idx: int, source):
        # open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Camera[{cam_idx}] unable to open source: {source}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_path = os.path.join(OUTPUT_DIR, f"cam{cam_idx}_out.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"🎥 Camera[{cam_idx}] started -> output {out_path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # run yolo.track on this frame
                results = self.yolo.track(frame, persist=True, verbose=False,
                                          conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
                if results is None or len(results)==0 or results[0].boxes is None:
                    out.write(frame)
                else:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    if ids is None:
                        ids = [None] * len(boxes)
                    for bbox, cls_idx, track_id in zip(boxes, classes, ids):
                        cls_name = self.yolo.names[int(cls_idx)]
                        if cls_name != "person" or track_id is None:
                            continue
                        track_id = int(track_id)
                        # --- extract face from bbox region first (more reliable) ---
                        x1,y1,x2,y2 = map(int, bbox)
                        x1, x2 = max(0,x1), min(frame.shape[1]-1, x2)
                        y1, y2 = max(0,y1), min(frame.shape[0]-1, y2)
                        crop = frame[y1:y2, x1:x2]
                        face_name = None
                        face_dist = 1e6
                        # Try on whole crop first
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        face_emb = extract_face_embedding_from_crop(crop_rgb)
                        # If face not found inside full body crop, attempt face detection on the full frame region slightly above bbox
                        if face_emb is None:
                            # attempt a small head region: top 40% of bbox
                            head_y2 = int(y1 + 0.45 * (y2 - y1))
                            if head_y2 > y1:
                                head_crop = frame[y1:head_y2, x1:x2]
                                head_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                                face_emb = extract_face_embedding_from_crop(head_rgb)
                        # If face embedding found, match against face_memory
                        with self.lock:
                            if face_emb is not None and len(self.face_memory)>0:
                                fname, dist = find_best_face_match(face_emb, self.face_memory)
                                face_name, face_dist = fname, dist
                        if face_emb is not None and face_name is not None and face_dist <= FACE_MATCH_THRESHOLD:
                            name = face_name
                            # update face avg
                            with self.lock:
                                # update face memory in-place & pending updates
                                for i,(n,e) in enumerate(self.face_memory):
                                    if n == name:
                                        new_avg = (UPDATE_ALPHA * e) + ((1.0-UPDATE_ALPHA) * face_emb)
                                        new_avg = new_avg / (np.linalg.norm(new_avg)+1e-12)
                                        self.face_memory[i] = (n, new_avg.astype(np.float32))
                                        self.pending_face_updates[n] = new_avg.astype(np.float32)
                                        break
                                else:
                                    # new face
                                    norm = face_emb / (np.linalg.norm(face_emb)+1e-12)
                                    self.face_memory.append((name, norm.astype(np.float32)))
                                    self.pending_face_updates[name] = norm.astype(np.float32)
                                    self.used_names.add(name)
                            # assign name for this track
                            self.assigned_names[(cam_idx,track_id)] = name
                            label = f"{name} (face)"
                        else:
                            # fallback: body embedding
                            emb = extract_body_embedding(frame, bbox)
                            if emb is None:
                                continue
                            with self.lock:
                                bname, sim = find_best_body_match(emb, self.body_memory) if self.body_memory else (None, -1.0)
                            if bname is not None and sim >= MATCH_THRESHOLD:
                                name = bname
                                # update averaged body embedding
                                with self.lock:
                                    for i,(n,e) in enumerate(self.body_memory):
                                        if n == name:
                                            new_avg = (UPDATE_ALPHA * e) + ((1.0 - UPDATE_ALPHA) * emb)
                                            new_avg = new_avg / (np.linalg.norm(new_avg)+1e-12)
                                            self.body_memory[i] = (n, new_avg.astype(np.float32))
                                            self.pending_body_updates[n] = new_avg.astype(np.float32)
                                            break
                                self.assigned_names[(cam_idx,track_id)] = name
                                label = f"{name} (body {sim:.2f})"
                            else:
                                # New person: assign next available name or Unknown_N
                                with self.lock:
                                    if self.next_name_idx < len(self.name_list):
                                        name = self.name_list[self.next_name_idx]
                                        self.next_name_idx += 1
                                    else:
                                        kne = 0
                                        while True:
                                            cand = f"Unknown_{kne}"
                                            if cand not in self.used_names:
                                                name = cand
                                                break
                                            kne += 1
                                    # store normalized emb
                                    emb_norm = emb / (np.linalg.norm(emb)+1e-12)
                                    self.body_memory.append((name, emb_norm.astype(np.float32)))
                                    self.pending_body_updates[name] = emb_norm.astype(np.float32)
                                    self.used_names.add(name)
                                    self.assigned_names[(cam_idx, track_id)] = name
                                    label = f"{name} (new)"
                        # draw
                        x1c,y1c,x2c,y2c = x1,y1,x2,y2
                        cv2.rectangle(frame, (x1c,y1c),(x2c,y2c),(0,200,0),2)
                        display_name = self.assigned_names.get((cam_idx,track_id), "...")
                        cv2.putText(frame, display_name, (x1c, max(15,y1c-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0),2)
                    out.write(frame)
                # preview small window optional: show per cam if desired (disabled in headless)
                cv2.imshow(f"Cam{cam_idx}", cv2.resize(frame,(640,360)))
                self.frame_counter += 1
                # periodic flush
                if self.frame_counter % BATCH_WRITE_INTERVAL == 0 and (self.pending_body_updates or self.pending_face_updates):
                    with self.lock:
                        try:
                            if self.pending_body_updates:
                                bulk_update_body(self.pending_body_updates)
                                self.pending_body_updates.clear()
                            if self.pending_face_updates:
                                bulk_update_faces(self.pending_face_updates)
                                self.pending_face_updates.clear()
                            # reload memories for consistency
                            self.body_memory = load_all_body_embeddings()
                            self.face_memory = load_all_face_embeddings()
                            print(f"💾 [cam{cam_idx}] flushed DB at frame {self.frame_counter}")
                        except Exception as e:
                            print("⚠️ DB flush error:", e)
                # allow quitting by pressing 'q' in any spawned window (if GUI present)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            out.release()

    def start(self):
        futures = []
        for idx, src in enumerate(self.camera_sources):
            futures.append(self.executor.submit(self.process_camera, idx, src))
        # wait for all to complete
        try:
            for f in futures:
                f.result()
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
        finally:
            # final flush
            with self.lock:
                try:
                    if self.pending_body_updates:
                        bulk_update_body(self.pending_body_updates)
                        self.pending_body_updates.clear()
                    if self.pending_face_updates:
                        bulk_update_faces(self.pending_face_updates)
                        self.pending_face_updates.clear()
                    print("💾 Final flush complete.")
                except Exception as e:
                    print("⚠️ Final DB flush error:", e)

# ---------------- UTIL: youtube download if needed ----------------
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

# ---------------- MAIN ----------------
def main():
    # Prepare DBs and build gallery faces into face DB
    init_dbs()
    print("📚 Building face gallery from", GALLERY_DIR)
    gallery_faces = build_face_gallery(GALLERY_DIR)
    # Optionally download test video to use as a source if you only have one camera and want to simulate
    # video_path = download_youtube(YOUTUBE_URL, INPUT_FILE)
    # just ensure CAMERA_SOURCES configured by user; can include video_path(s)

    mgr = MultiCamReID(CAMERA_SOURCES, YOLO_WEIGHTS)
    # reload face memory (gallery persisted)
    mgr.face_memory = load_all_face_embeddings()
    mgr.body_memory = load_all_body_embeddings()
    print(f"🔁 Starting multi-camera ReID for {len(CAMERA_SOURCES)} sources.")
    mgr.start()
    print("✅ All cameras finished. Outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
