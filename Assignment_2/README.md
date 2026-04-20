# On-Device Keyword Spotting using MFCC and TinyML

## 📌 Project Overview
This project implements a **real-time keyword spotting system** on an embedded platform using audio data acquired from a **microphone connected to an Arduino-compatible board**. The system performs **feature extraction using Mel-Frequency Cepstral Coefficients (MFCC)** and executes **inference directly on the device**.

The full pipeline includes **audio data collection**, **MFCC feature extraction using CMSIS DSP**, **model training in Google Colab**, and **deployment of the trained model on Arduino for on-device inference**.

## 🎤 Audio Data Collection

### 🔊 Supported audio classes

- Clap  
- Tap  
- Snap  
- Silence  

Audio samples are recorded using a microphone and segmented into short frames for processing.

## 🎚️ MFCC Feature Extraction Pipeline

The MFCC implementation follows a standard DSP pipeline adapted for embedded systems using the **CMSIS DSP framework**.

### 1. Audio Sampling
The microphone signal is sampled at a fixed frequency and stored in a buffer.

### 2. Frame Segmentation
The continuous audio stream is divided into overlapping frames of fixed length.

### 3. Windowing (Hamming Window)
A window function is applied to each frame to reduce spectral leakage.

### 4. RFFT (Real Fast Fourier Transform)
The signal is transformed from the time domain to the frequency domain using CMSIS DSP FFT functions.

### 5. Mel Filter Bank Processing
The frequency spectrum is mapped onto the Mel scale using a bank of triangular filters.

### 6. Log Energy Computation
The logarithm of the Mel filter outputs is computed to simulate human perception of sound.

### 7. Discrete Cosine Transform (DCT)
The DCT is applied to decorrelate the features and produce the MFCC coefficients.

### 8. MFCC Coefficients
The resulting coefficients form the feature vector used for classification.

## 💪 Model Training in Google Colab
We collect and organize audio samples for each class, extract MFCC features, and use them to train a lightweight neural network.

The training pipeline includes:

- Loading and preprocessing the dataset  
- MFCC extraction in Python (for validation against embedded implementation)  
- Training a **Convolutional Neural Network (CNN)**  
- Evaluating performance using **accuracy and confusion matrix**  
- Exporting the trained model parameters for embedded deployment  

## 💻 Deployment of the Trained Model on Arduino
The trained model is deployed on the Arduino board for **real-time inference**.

The embedded system performs:

- Real-time audio acquisition from the microphone  
- On-device MFCC computation using CMSIS DSP  
- Feature normalization consistent with training  
- Execution of the neural network inference  
- Output of the predicted class via the Serial Monitor  

The implementation is optimized to meet **memory and timing constraints** of embedded systems.

## ⚙️ Instructions

- Store all recorded audio samples in the `data/` folder.  
- Open the Colab notebook `training.ipynb`, upload the dataset, and run all cells to train the model.  
- Export the trained model (e.g., as a `.tflite` file and corresponding header file).  
- Upload the Arduino sketch (including the model file and MFCC implementation) to the board.  
- Open the Serial Monitor and test real-time keyword recognition.  

---

Ilka BRETSCHNEIDER (268664)<br> Romain NOBLET (268709)

*Università degli Studi di Trento - 2026*