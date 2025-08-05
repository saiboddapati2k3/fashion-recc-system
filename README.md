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

## 🖼️ Demo

Upload any fashion product image (like a shoe, t-shirt, or dress), and the app returns **5 most visually similar items** from the dataset.

> 📷 Screenshots can be added here after running the app

---

## 📁 Dataset

This project uses the **Fashion Product Images (Small)** dataset from Kaggle:  
🔗 [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

> ⚠️ **Note:** You must download the dataset manually from Kaggle before running the app.

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


---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/fashion-recommender.git
   cd fashion-recommender

Create a virtual environment

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
Install the dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download and prepare the dataset

Download from Kaggle

Place all images in the images/ folder

Place styles.csv in the root directory

Generate image embeddings and FAISS index

bash
Copy
Edit
python embeddings.py
Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
🧠 Tech Stack
Python 3.10+

PyTorch – for ResNet50 model and feature extraction

FAISS – for fast vector similarity search

Streamlit – frontend web interface

Pandas, NumPy, Pillow – for image and metadata handling

⏱ Performance
~44,000 fashion product images

~5 ms retrieval time per query

Embedding size: 2048-d

Index type: FAISS IndexFlatL2

🧩 Future Enhancements
 Add CLIP support for text + image recommendations

 Add filters (category, color, gender, etc.)

 Include product details in the UI

 Deploy to Hugging Face Spaces or Streamlit Cloud

👨‍💻 Author
Sai Boddapati
Built as a project to demonstrate practical skills in deep learning, computer vision, and recommender systems.

📄 License
This project is licensed under the MIT License – feel free to use and modify.