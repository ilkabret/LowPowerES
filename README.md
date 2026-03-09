# 🤖 Gesture Recognition Application Using Feature Extraction

# 🎯 Project Overview

This project implements a **real-time gesture recognition system** using motion sensor data from an **Arduino-compatible IMU** (accelerometer and/or gyroscope). The system uses feature extraction techniques such as **RMS** and **PSD** to classify multiple gesture types and performs **inference directly on the device**.

The full pipeline includes **data collection**, **feature extraction**, **model training in Google Colab**, and **deployment of the trained model on Arduino for on-device inference**.

---

# ✋ Gesture Classes

The system recognizes multiple gesture classes, for example:

- 🧘 Rest  
- 👋 Shake (left-right)  
- ⬆️⬇️ Up-Down  
- 🔄 Circle  
- …  

Additional gesture classes can be added by collecting more labeled data.

---

# ⚙️ Feature Extraction

Features are extracted using **sliding windows** over the signal.

## 🕒 Time-Domain Features

- 📊 Mean  
- 📉 Standard Deviation  
- 📈 RMS (Root Mean Square)  
- 🔽 Minimum and 🔼 Maximum  

## 🌊 Frequency-Domain Features

- 📡 Power Spectral Density (PSD)

These features are computed for the **accelerometer and gyroscope axes**.

---

# 🧠 Model Training (Google Colab)

The training pipeline includes:

- 🔄 Data normalization  
- 🤖 Training a NN  
- 📊 Model evaluation using accuracy and a confusion matrix  

The trained model parameters (**weights and biases**) are exported for deployment on Arduino.

---

# 🔌 On-Device Inference (Arduino)

The Arduino sketch performs:

- 📡 Real-time IMU data acquisition  
- 🪟 Window buffering  
- ⚙️ Feature computation (same features used during training)  
- 🔄 Feature normalization  
- 🧠 Gesture classification using exported model parameters  
- 🖥️ Output of predicted gesture via the Serial monitor  

This enables **real-time gesture recognition directly on the embedded device**.

---

Ilka BRETSCHNEIDER
Romain NOBLET

**Università degli Studi di Trento**
