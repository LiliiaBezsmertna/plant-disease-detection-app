# 🌿 Plant Disease Detection App

This project is a deep learning application that detects whether a plant leaf is **Healthy** or **Diseased** using a CNN model (MobileNetV2 with transfer learning).

## 🚀 Features

* Image upload or camera input
* Automatic preprocessing (crop + resize)
* Binary classification (Healthy / Disease)
* Confidence score visualization
* Streamlit interactive interface

## 🧠 Model

Architecture:

* MobileNetV2 (pretrained on ImageNet)
* Transfer learning
* Data augmentation
* Binary classification layer (sigmoid)

Input size:
224 × 224 pixels

Validation accuracy:
~97–98%

## 📊 Business Value

This solution can support:

* early detection of plant diseases
* reduction of manual inspection work
* improved crop quality
* smart agriculture automation

Example usage:

* greenhouses
* farms
* food production companies
* agricultural monitoring systems

## ⚠️ Challenges and Limitations

Possible challenges include:

* model confidence depends on image quality
* dataset bias may affect predictions
* requires diverse training images for scalability
* should not replace expert diagnostics in critical cases

## 🔐 Ethical Considerations

Responsible usage requires:

* transparency about prediction uncertainty
* human validation in production environments
* continuous dataset improvement
* avoidance of automated decision-making without supervision

## 🛠 Technologies Used

Python
TensorFlow / Keras
Streamlit
NumPy
Pillow

## ▶️ Run the App

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

## 📁 Project Structure

plant_disease_app/
│
├── app.py
├── train_model.py
├── plant_disease_model.keras
├── requirements.txt
├── dataset/ (training images)
└── README.md
