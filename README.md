# 🚦 Auto Rickshaw Detection System

📄 **Preprint (arXiv):**  
**Auto Rickshaw Detection in the Wild**  
👉 [https://arxiv.org/abs/2510.26154](https://arxiv.org/abs/2510.26154)

---

## 🔍 Overview

This project uses **YOLOv8** for detecting and classifying **auto rickshaws** and **non-auto rickshaws** in images. It is particularly tailored for **South Asian road environments** and real-world challenges such as **occlusion**, **class imbalance**, and **crowded traffic scenes**. The system focuses on handling dense urban traffic environments, making it ideal for **traffic analytics** and **smart city applications**.

Unlike generic vehicle detection models, this system focuses on **auto rickshaws**, addressing real-world challenges that occur in **dense traffic**. It is built to detect **auto rickshaws** even when partially occluded by other vehicles or pedestrians.

---

## 🧠 Key Challenges Addressed

- **Occlusion:** Auto rickshaws are often partially hidden by buses, cars, or pedestrians.
- **Class Imbalance:** Auto rickshaws appear much less frequently than other vehicles.
- **Dense Urban Traffic:** Overlapping objects and cluttered scenes.

These challenges are handled through **dataset curation**, **annotation strategy**, **data augmentation**, and **careful YOLO training design**.

---

## ✨ Features

- Detects **auto rickshaws** (engine-powered) and **non-auto rickshaws** (manually paddled)
- Robust to **occlusion** and **dense traffic**
- Supports **multiple detections per image**
- Outputs **confidence scores** and **bounding boxes**
- Lightweight inference suitable for **real-time deployment**
- Visualizes detection results with **color-coded bounding boxes** (green for auto, red for non-auto)

## 🖼️ Sample Results

<p align="center">
  <img src="assest/sample1.png" width="30%" />
  <img src="assest/sample2.png" width="30%" />
  <img src="assest/sample3.png" width="30%" />
</p>

---

## Setup

1. Install the required dependencies:
   pip install -r requirements.txt

3. Dataset structure:
   - The system uses labeled images from the rickshaw_labeled_images folder
   - Images are stored in the rickshaw_labeled_images/images directory
   - Labels are stored in the rickshaw_labeled_images/labels directory in YOLO format

```bash
pip install -r requirements.txt
