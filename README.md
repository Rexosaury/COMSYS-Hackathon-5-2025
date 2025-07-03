# COMSYS-Hackathon-5-2025

**Robust Face Recognition & Gender Classification under Adverse Conditions**

- ✅ **Task A:** Gender Classification using MobileNetV2
- ✅ **Task B:** Robust Face Recognition using InsightFace ArcFace Embeddings + SVM
- ✅ **Dataset:** FACECOM (faces under adverse conditions)

---

## 📌 Project Highlights

- Designed for **real-world scenarios** with image distortions: sunny, rainy, blurred, noisy conditions.
- Uses **transfer learning** for gender classification with class imbalance handling.
- Uses **embedding-based face recognition** with ArcFace + SVM for identity matching.
- Visualizes results: accuracy, confusion matrix, per-class precision & recall.

---

## 🗂️ Repository Structure


- **├── Task_A/**
- **│ ├── Gender_classfication.py/** 
- **├── Task_B/**
- **│ ├── Task_B_InsightFace.ipynb/**

	
---

## 📄 Documentation

* 📌 **Architecture Diagram:** [`docs/Model_Architecture(Task_A).jpeg`](docs/Model_Architecture(Task_A).jpeg) 
[`docs/Model_Architecture(Task_B).jpeg`](docs/Model_Architecture(Task_B).jpeg) 

* 📌 **Technical Summary:** [`docs/Technical_summary.pdf`](docs/Technical_summary.pdf)

---


## ✅ Task A: Gender Classification

**Model:**  
- Pre-trained **MobileNetV2** backbone.
- Final dense layer for binary classification.
- Uses `ImageDataGenerator` for data augmentation.
- Splits dataset **70% train / 30% val** dynamically.
- Handles class imbalance with `compute_class_weight`.

**Key Steps:**
1. Split raw dataset into balanced train/val folders.
2. Train MobileNetV2 with frozen base.
3. Save `.keras` model to Google Drive.
4. Load & evaluate on validation set.
5. Visualize confusion matrix & classification report.

**Expected Accuracy:** ~90–95% with good quality balanced data.

---

## ✅ Task B: Face Recognition

**Model:**  
- Uses **InsightFace** ArcFace (`buffalo_l`).
- Generates normalized embeddings per face.
- Averages embeddings per identity for robust matching.
- Classifies new faces using **SVM** with cosine similarity.
- Includes **image denoising & filtering** for distorted inputs.
- Uses **GridSearchCV** to tune SVM parameters for better accuracy.

**Key Steps:**
1. Extract embeddings for all `train/` images (clear + distortions).
2. Apply CLAHE + denoising filters to distortions to enhance clarity.
3. Average multiple embeddings per identity.
4. Train & tune SVM.
5. Predict on `val/` images.
6. Visualize predictions with confidence score.

**Expected Accuracy:** ~85–95% if good frontal images are used.

---

## ⚙️ How to Run 

**1️⃣ Install Dependencies**
```bash
pip install insightface onnxruntime-gpu opencv-python-headless scikit-learn matplotlib tqdm tensorflow
```

**2️⃣ Mount Google Drive**
```bash
from google.colab import drive
drive.mount('/content/drive')
```
**3️⃣ Run Scripts
**For Task A:
```bash
python juclassifydiff.py
```
**For Task B:
```bash
Open Task_B_InsightFace.ipynb in Colab → Run all cells.
```

**📈 Results 

| Task                     |  Accuracy  | Precision | Recall | F1-Score |
| ------------------------ | :--------: | :-------: | :----: | :------: |
| Gender Classification    |   0.91     |   0.91    | 0.91   |  0.91    |
| Face Recognition         |   0.0000   |   0.0000  | 0.0000 |  0.0000  |
| **Final Weighted Score** | **0.0000** |     —     |    —   |     —    |

----
## 👥 Team

* **Team Leader:** Sobhan Roy
* **Team Members:** Annick Das, Suchismita Bakshi
* **Affiliation:** Techno International Newtown
* **Contact:** [roysobhan.sr@gmail.com](mailto:roysobhan.sr@gmail.com)

-----
## 📜 License

This project was created for COMSYS Hackathon-5 2025 and is solely intended for educational and research purposes.

---

```



