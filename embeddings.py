#!/usr/bin/env python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   

import pathlib, time, sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch
from torchvision import models, transforms
import faiss

# Paths
ROOT        = pathlib.Path(__file__).resolve().parent       
IMG_DIR     = ROOT / "images"
META_CSV    = ROOT / "styles_clean.csv"
OUT_EMB_NPY = ROOT / "embeddings.npy"
OUT_FAISS   = ROOT / "faiss_index.bin"
OUT_META    = ROOT / "metadata.parquet"

DEVICE = "cpu" 
BATCH_SIZE  = 32      


# 1. Load cleaned metadata 
print("🔹 Loading metadata …")
df = pd.read_csv(META_CSV)
n_items = len(df)
print(f"   {n_items:,} rows")

# 2. Prepare ResNet-50 backbone 
print("🔹 Loading ResNet-50 weights …")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # drop classification layer -> 2048-dim
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Helper: load & preprocess batch 
def load_batch(start, end):
    imgs = []
    for img_id in df["id"].iloc[start:end]:
        path = IMG_DIR / f"{img_id}.jpg"
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] {path.name}: {e}", file=sys.stderr)
            img = Image.new("RGB", (224, 224), (0, 0, 0))   
        imgs.append(transform(img))
    return torch.stack(imgs).to(DEVICE)   

# 4. Extract embeddings in batches 
print("Extracting embeddings …")
emb_matrix = np.zeros((n_items, 2048), dtype="float32")

with torch.no_grad():
    for start in tqdm(range(0, n_items, BATCH_SIZE)):
        end   = min(start + BATCH_SIZE, n_items)
        batch = load_batch(start, end)
        vecs  = model(batch).cpu().numpy()
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        emb_matrix[start:end] = vecs

# 5. Build FAISS index 
print("Building FAISS index …")
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)


print("Saving files …")
np.save(OUT_EMB_NPY, emb_matrix)
faiss.write_index(index, str(OUT_FAISS))
df.to_parquet(OUT_META, index=False)

print("\n Finished ")
print(f"   {OUT_EMB_NPY.name}  |  shape {emb_matrix.shape}")
print(f"   {OUT_FAISS.name}    |  {index.ntotal:,} vectors")
print(f"   {OUT_META.name}     |  {len(df):,} rows")
