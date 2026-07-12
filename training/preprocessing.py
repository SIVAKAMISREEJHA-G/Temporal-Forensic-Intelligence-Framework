import os, glob, json, cv2, numpy as np, torch, torchvision.transforms as T
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR     = r"C:\Users\sivak\Downloads\crime data"
PROCESSED_DIR = os.path.join(BASE_DIR, "TFIF", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

CLASSES = ["Abuse","Car Accident","Explosion","Fighting","Normal","Riot","Shooting"]
LABEL_MAP = {c:i for i,c in enumerate(CLASSES)}
NUM_FRAMES = 16
IMG_SIZE   = 224
EXTS       = ["*.mp4","*.avi","*.mkv","*.mov"]

# ── MobileNetV3 backbone (frozen) ──────────────────────────────────────────
weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
backbone = mobilenet_v3_large(weights=weights)
backbone.classifier = torch.nn.Identity()   # outputs 960-dim vector
backbone.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def extract_frames(video_path, n_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1: total = 1
    indices = np.linspace(0, total-1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frm = cap.read()
        if not ret:
            frm = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        frames.append(frm)
    cap.release()
    return frames

def embed_video(video_path):
    frames = extract_frames(video_path)
    tensors = []
    with torch.no_grad():
        for frm in frames:
            img = Image.fromarray(frm)
            t = transform(img).unsqueeze(0)        # (1,3,H,W)
            feat = backbone(t)                      # (1,960)
            tensors.append(feat.squeeze(0))
    seq = torch.stack(tensors)                     # (16,960)
    return seq

records = []
print("Extracting frame embeddings — this may take a while on CPU …")
for cls in CLASSES:
    cls_dir = os.path.join(BASE_DIR, cls)
    if not os.path.exists(cls_dir):
        print(f"  [SKIP] {cls_dir} not found"); continue
    videos = []
    for ext in EXTS:
        videos.extend(glob.glob(os.path.join(cls_dir, ext)))
    print(f"  {cls}: {len(videos)} videos")
    for vp in videos:
        vname = os.path.basename(vp)
        safe  = vname.replace(" ","_").replace("(","").replace(")","").replace(".mp4","")
        out_path = os.path.join(PROCESSED_DIR, f"{cls}_{safe}.pt")
        if not os.path.exists(out_path):
            try:
                seq = embed_video(vp)
                torch.save(seq, out_path)
            except Exception as e:
                print(f"    [ERR] {vname}: {e}"); continue
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        records.append({"class":cls,"label":LABEL_MAP[cls],"video":vname,"feature_path":out_path,"duration":round(fc/fps,2),"fps":fps})

# -- save metadata ----------------------------------------------------------
meta_path = os.path.join(BASE_DIR,"TFIF","data","metadata.json")
with open(meta_path,"w") as f:
    json.dump(records, f, indent=2)
print(f"Metadata saved ({len(records)} entries) -> {meta_path}")

# -- stratified splits ------------------------------------------------------
labels = [r["label"] for r in records]
idx_all = list(range(len(records)))
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(sss1.split(idx_all, labels))
temp_labels = [labels[i] for i in temp_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_rel, test_rel = next(sss2.split(temp_idx, temp_labels))
val_idx  = [temp_idx[i] for i in val_rel]
test_idx = [temp_idx[i] for i in test_rel]

splits = {"train":list(map(int,train_idx)),"val":list(map(int,val_idx)),"test":list(map(int,test_idx))}
splits_path = os.path.join(BASE_DIR,"TFIF","data","splits.json")
with open(splits_path,"w") as f:
    json.dump(splits, f, indent=2)
print(f"Splits saved -> {splits_path}  (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

label_map_path = os.path.join(BASE_DIR,"TFIF","model","label_map.json")
os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
with open(label_map_path,"w") as f:
    json.dump({"classes":CLASSES,"label_map":LABEL_MAP}, f, indent=2)
print("label_map.json saved.")
