#!/usr/bin/env python
import pathlib, io
from functools import lru_cache
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms

st.set_page_config(page_title="Fashion Recommender", layout="wide")


# ─── Paths ────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent
IMG_DIR     = ROOT / "images"
EMB_NPY     = ROOT / "embeddings.npy"
FAISS_BIN   = ROOT / "faiss_index.bin"
META_PARQ   = ROOT / "metadata.parquet"

# ─── Cached lazy-loaders  (FAISS import happens *here*) ───────────────────
@st.cache_resource(show_spinner="🔹 Loading FAISS index …")
def get_faiss_index():
    import faiss  # local import avoids early double-import
    return faiss.read_index(str(FAISS_BIN))

@st.cache_resource(show_spinner="🔹 Loading embeddings …")
def get_embeddings() -> np.ndarray:
    return np.load(EMB_NPY)

@st.cache_resource(show_spinner="🔹 Loading metadata …")
def get_meta() -> pd.DataFrame:
    return pd.read_parquet(META_PARQ)

@st.cache_resource(show_spinner="🔹 Loading ResNet-50 …")
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to("cpu")
    return model

# ─── Image → 2048-D feature ───────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
def embed(img: Image.Image) -> np.ndarray:
    t = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        v = get_model()(t).squeeze(0).numpy()
    return (v / (np.linalg.norm(v)+1e-12)).astype("float32")

# ─── UI ───────────────────────────────────────────────────────────────────
st.title("👗  Fashion Image Recommender")
k = st.sidebar.slider("Number of results", 3, 12, 5)

upl = st.file_uploader("Upload a product photo", type=["jpg","jpeg","png"])
if upl:
    q_img = Image.open(upl).convert("RGB")
    st.image(q_img, caption="Query", width=300)

    with st.spinner("Searching …"):
        q_vec = embed(q_img)[None, :]
        D,I   = get_faiss_index().search(q_vec, k)

    meta = get_meta().iloc[I[0]]
    cols = st.columns(k)
    for col, dist, row in zip(cols, D[0], meta.itertuples()):
        img_p = IMG_DIR / f"{row.id}.jpg"
        if img_p.exists():
            col.image(str(img_p), use_column_width=True)
        col.caption(
            f"**{row.productDisplayName}**\n"
            f"*{row.articleType}* · {row.baseColour}\n"
            f"distance {dist:.3f}"
        )
else:
    st.info("⬆️  Upload an image to start")
