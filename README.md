# Deepfake
# 🧠 Deepfake Detection System

A deep learning-based web application that detects whether an image is **REAL or FAKE (Deepfake)** using a Convolutional Neural Network (CNN) built with PyTorch and deployed via Flask.

---

## 🚀 Features

* 🔍 Detects deepfake vs real images
* ⚡ Fast inference using trained CNN model
* 🌐 Simple web interface (Flask)
* 🧠 GPU support (CUDA enabled)
* 📊 Confidence score output
* 🔒 Secure image upload handling

---

## 🏗️ Project Structure

```
deepfake-project/
│
├── model.py                # CNN model architecture
├── train.py                # Training script
├── app.py                  # Flask web app
├── model.pth               # Trained model weights
│
├── templates/
│   └── index.html          # Frontend UI
│
├── uploads/                # Uploaded images
│
└── archive/                # Dataset (real & fake images)
```

---

## 🧪 Dataset

The dataset consists of labeled images:

* **Real Images → Label 0**
* **Fake Images → Label 1**

Dataset is automatically scanned using folder structure:

```
archive/
   ├── real/
   └── fake/
```

---

## 🏋️ Model Training

The model is trained using:

* **Framework:** PyTorch
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam
* **Image Size:** 224×224
* **Epochs:** 5 (configurable)

### 🔧 Data Preprocessing

* Resize
* Normalization
* Data Augmentation (flip, rotation, jitter)

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/06K07/Deepfake.git
cd deepfake-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python train.py
```

### 4️⃣ Run Web App

```bash
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🖼️ Usage

1. Upload an image (JPG/PNG)
2. Click **Detect**
3. Get result:

   * ✅ REAL
   * ❌ FAKE
   * 📊 Confidence Score

---

## ⚠️ Limitations

* Works only on **images (not videos)**
* Accuracy depends on dataset quality
* May struggle with unseen manipulation techniques

---

## 🔮 Future Improvements

* 🎥 Video deepfake detection
* 📱 Mobile app integration
* ☁️ Cloud deployment (AWS/Render)
* 🧠 Use advanced models (ResNet, EfficientNet)
* 📊 Visualization dashboard

---

## 🛠️ Tech Stack

* Python
* PyTorch
* OpenCV
* Flask
* HTML/CSS

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Submit a pull request

---

## 📜 License

This project is for educational purposes.

---

## 👨‍💻 Author

**Krishan Kumar,Lovish Baweja & Yusuf**
AI/Ml & Cybersecurity Enthusiast

---

## ⭐ Acknowledgements

* PyTorch community
* Open-source datasets
* Research on deepfake detection

---
