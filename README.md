#  Fashion Image Recommendation System

A **content-based fashion image recommendation system** that retrieves visually similar products using deep learning and FAISS. Upload a fashion image and get top-5 visually similar results in real-time via a **Streamlit web app**.

---

## 🚀 Features

- Extracts **2048-dimensional embeddings** using a pretrained **ResNet50** model (PyTorch)
- Uses **ℓ2-normalized embeddings** for accurate similarity comparison
- Performs top-5 image recommendation using **FAISS** (Facebook AI Similarity Search)
- Fast, lightweight, and deployable on **CPU**
- Built with **Streamlit** for a smooth UI

---

### Place the files as follows:


fashion-recc-system/
├── app.py
├── embeddings.py
├── images/ <- place all .jpg images here
├── styles.csv <- original metadata file
├── embeddings.npy <- generated after running embeddings.py
├── faiss_index.bin <- generated after running embeddings.py
├── metadata.parquet <- saved metadata file
├── requirements.txt
└── README.md


------

## 🛠️ Installation Guide

### ✅ 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### ✅ 2. Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ 3. Download and prepare the dataset

- Download from Kaggle:  
  [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

- Place files as follows:
  - All image files → `images/` folder
  - `styles.csv` → project root

---

## 🔍 Generate Embeddings & Build Index

Run the following script to extract image embeddings and build the FAISS index:

```bash
python embeddings.py
```

---

## 🚀 Run the Streamlit App

Start the interactive web app:

```bash
streamlit run app.py
```

---

## 🧠 Tech Stack

- **Python 3.10+**
- **PyTorch** – ResNet50 for feature extraction
- **FAISS** – Efficient similarity search
- **Streamlit** – Lightweight frontend UI
- **NumPy, Pandas, Pillow** – Data processing and image loading

---

## ⏱️ Performance

- 📦 ~**44,000** fashion product images  
- ⚡ ~**5 ms** retrieval time per image query  
- 📐 Embedding size: **2048 dimensions**  
- 🔎 Index type: `FAISS IndexFlatL2`

---

## 🔮 Future Enhancements

- [ ] Add **CLIP** support for **multimodal (image + text)** recommendations
- [ ] Implement advanced **filtering** by category, gender, color, etc.
- [ ] Include **product details** in search results
- [ ] Deploy to **Hugging Face Spaces** or **Streamlit Cloud**

---

## 👨‍💻 Author

**Sai Boddapati**  
Built as a showcase project in deep learning, computer vision, and recommender systems.

---

## 📄 License

Licensed under the MIT License – feel free to use and modify.