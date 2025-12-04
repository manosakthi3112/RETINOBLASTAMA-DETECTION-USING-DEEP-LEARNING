# Retinoblastoma Detection System using EfficientNetV2-M & Explainable AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-orange)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An automated, deep-learning-based diagnostic system designed to detect **Retinoblastoma** (a life-threatening pediatric eye cancer) from retinal fundus images. This project utilizes the **EfficientNetV2-M** architecture for high-accuracy classification and integrates **Explainable AI (Grad-CAM)** to provide visual heatmaps, ensuring clinical interpretability.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology & Pipeline](#-methodology--pipeline)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [References](#-references)

---

## 📌 Overview

Retinoblastoma is a rare but serious cancer of the retina affecting young children (mostly under age 5). Early detection is critical for saving vision and life. However, manual screening is subjective, time-consuming, and prone to error due to the rarity of the disease.

**The Solution:**
This project provides an automated screening tool that:
1.  **Enhances** poor-quality pediatric fundus images.
2.  **Classifies** images as *Normal* or *Disease* (Retinoblastoma).
3.  **Explains** the decision by highlighting tumor regions using heatmaps, helping doctors trust the AI.

---

## 🚀 Key Features

*   **Multi-Stage Preprocessing:** A specific pipeline designed for retinal images:
    *   **MSRCR:** Multi-Scale Retinex with Color Restoration to fix uneven lighting.
    *   **CLAHE:** Contrast-Limited Adaptive Histogram Equalization for local contrast.
    *   **Vessel Suppression:** Morphological operations to remove blood vessel noise.
*   **EfficientNetV2-M Backbone:** Utilizes Transfer Learning (pretrained on ImageNet) for superior accuracy and parameter efficiency compared to VGG or ResNet.
*   **Explainable AI (Grad-CAM):** Generates Gradient-weighted Class Activation Maps to visualize the specific anatomical regions (e.g., calcifications, mass lesions) driving the prediction.
*   **Web-Based Interface:** A Flask application allowing clinicians to upload scans and view instant results.
*   **Robust Training:** Implements data augmentation (rotation, flipping, color jitter) and early stopping.

---

## ⚙️ Methodology & Pipeline

The system processes data in four distinct stages:

1.  **Data Acquisition:** 
    *   Input: Retinal fundus images (RetCam).
    *   Format: Resized to 224x224 pixels.
2.  **Preprocessing Module:**
    *   Input images undergo MSRCR -> CLAHE -> Vessel Suppression.
3.  **Feature Extraction:**
    *   **EfficientNetV2-M** extracts hierarchical features.
    *   Initial layers capture edges; deep layers capture tumor morphology.
4.  **Classification & Visualization:**
    *   **Binary Classifier:** Sigmoid output (Probability > 0.5 = Disease).
    *   **Grad-CAM:** Overlays a heatmap on the original image.

---

## 🏗 System Architecture

**Hardware Requirements:**
*   **GPU:** NVIDIA GeForce GTX Titan (or equivalent) recommended for training.
*   **RAM:** 8GB Minimum.
*   **Storage:** ~10GB for datasets and model weights.

**Software Architecture:**
The solution is built on Python, using PyTorch for the backend logic and Flask for the frontend presentation.

---

## 💻 Tech Stack

*   **Programming Language:** Python 3.8+
*   **Deep Learning:** PyTorch 1.12, Torchvision
*   **Web Framework:** Flask
*   **Image Processing:** OpenCV (cv2), PIL, NumPy
*   **Data Handling:** Pandas
*   **Visualization:** Matplotlib, Seaborn
*   **Algorithms:** EfficientNetV2-M, Adam Optimizer, Binary Cross Entropy Loss

---

## 📂 Folder Structure

bash
Retinoblastoma-Detection/
│
├── app.py                   # Main Flask application for inference
├── training.py              # PyTorch training script
├── requirements.txt         # Dependencies
│
├── Final/                   # Dataset Directory
│   ├── Retinoblastoma/      # Positive images
│   └── Normal/              # Negative images
│
├── static/                  # Web assets
│   ├── uploads/             # User uploaded images
│   ├── results/             # Generated Grad-CAM results
│   ├── css/
│   └── js/
│
├── templates/               # HTML Views
│   ├── index.html           # Dashboard
│   └── architecture.html    # Model Visualization
│
├── models_1/                # Saved Model Weights (.pth)
└── README.md                # Documentation

#🔧 Installation

    
git clone https://github.com/yourusername/retinoblastoma-detection.git
cd retinoblastoma-detection

  

Create Virtual Environment
code Bash

    
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

  


Install Dependencies
code Bash

        
pip install torch torchvision opencv-python flask numpy matplotlib pandas tqdm

      

#🏃 Usage
1. Training the Model

To train the model on your own dataset, ensure your data is in the Final folder structure, then run:
code Bash

    
python training.py

  

The script will perform preprocessing, augmentation, and training for the specified epochs, saving the best model to the models_1 directory.
2. Running the Web App (Inference)

Once the model is trained (or using provided weights):

 Edit app.py to point to your saved model path:
 code Python

    
model_path = r"./models_1/best_model_epoch_X.pth"

  

Run the Flask app:
code Bash

        
 python app.py

      

 Open your browser at http://127.0.0.1:5000.

Upload a retinal image to see the classification and heatmap.

#📊 Results

The system evaluates images based on probability scores:

 Confidence Score: Displays the certainty of the model (e.g., 98.9%).

 Heatmap: Red areas indicate high importance (tumor regions), Blue areas indicate low importance.

Metrics used for evaluation:

 Accuracy, Precision, Recall, F1-Score.

 Dice Similarity Coefficient (DSC).

 Intersection over Union (IoU).

#📚 References

   EfficientNetV2: Tan, M., & Le, Q. V. (2021). "Smaller Models and Faster Training."

   Grad-CAM: Selvaraju, R. R., et al. (2017). "Visual Explanations from Deep Networks via Gradient-Based Localization."
 
   Retinex Theory: Rahman, Z., et al. (2016). "Retinex Theory Based Adaptive Filter for Color Correction."

#Disclaimer: This project is for educational and research purposes only. It is not a certified medical device and should not replace professional medical diagnosis.
