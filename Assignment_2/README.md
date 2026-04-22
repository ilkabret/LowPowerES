# On-Device Keyword Spotting using MFCC and TinyML

## 📌 Project Overview
This project implements a **real-time keyword spotting system** on an embedded platform using audio data acquired from an **Arduino-compatible microphone**. The system performs **feature extraction using Mel-Frequency Cepstral Coefficients (MFCC)** and performs **inference directly on the device**.

The full pipeline includes:
- **Audio data collection (for multiple audio classes)**
- **MFCC feature extraction using the CMSIS DSP framework**
- **Training of a classifier in Google Colab**
- **Deployment of the trained model on Arduino for on-device inference**

## 🎤 Audio Data Collection

Audio samples are recorded using an MP34DT06JTR microphone, which outputs a Pulse-Density Modulated (PDM) signal.
This signal is a binary signal stream, where the density of ones represents the amplitude of the audio waveform.

The MCU includes a PDM module that converts this stream into 16-bit Pulse-Code Modulated (PCM) samples.

### 🔊 Supported audio classes

- Clap  
- Tap  
- Snap  
- Whistle 

### 📊 Dataset Details

- Samples per class: 20
- Samples per recording: 2000 (representing 125ms @ 16kHz)
- Data format: 16-bit integer PCM values

### ⏱️ Sampling Configuration

The audio is sampled at 16 kHz.

The Nyquist–Shannon sampling theorem states that the sampling frequency must be at least twice the highest frequency present in the signal. -> avoid aliasing!

Since most relevant audio information for human-generated sounds is below 8 kHz, a 16 kHz sampling rate is sufficient to accurately capture the signal.

## 🎚️ MFCC feature extraction using the CMSIS DSP framework

The MFCC implementation follows the CMSIS structure discussed in class, using the **CMSIS DSP framework**.

### 1. Audio Sampling
The microphone signal is sampled and stored as stated in "Audio Data Collection".

### 2. Frame Segmentation
The continuous stream of samples is divided into short, overlapping frames of 256 samples each.
An overlap of 50% is used, meaning each sample appears in two consecutive frames. 
<!-- This improves temporal continuity and feature stability. -->

### 3. Windowing (Hamming Window)
Since each frame introduces sharp discontinuities at its boundaries, this can lead to spectral leakage in the frequency domain.
To mitigate this, a Hamming window is applied to each frame, smoothly tapering the signal toward zero at both ends.

### 4. RFFT (Real Fast Fourier Transform)
A Real FFT is used to transform each frame from the time domain to the frequency domain using CMSIS DSP functions.

Because the input signal is real-valued, the resulting spectrum is symmetric, and the second half does not contain additional information (mirror).
Therefore, only 129 unique frequency bins are retained (from an original 256-bins in FFT).

The power spectrum is then computed from the FFT output.

### 5. Mel Filter Bank Application
The Mel scale is a perceptual frequency scale in which equal distances correspond to equal perceived differences in pitch.

The frequency spectrum is mapped onto the Mel scale using a bank of triangular filters, emphasizing perceptually important frequency bands.

### 6. Log Energy Computation
The logarithm of the Mel filter bank energies is computed to compress the dynamic range of the features.

This step helps stabilize the input for the neural network and better reflects human loudness perception.

### 7. Discrete Cosine Transform (DCT)
Adjacent Mel filters have overlapping responses, which leads to correlated (redundant) features.

The Discrete Cosine Transform (DCT) decorrelates these values by transforming them into a set of approximately independent components, producing the MFCCs.

### 8. MFCC Coefficients
The resulting MFCC coefficients form the final feature vector used for classification.
- Lower-order coefficients capture the broad spectral envelope (overall shape of the sound)
- Higher-order coefficients capture finer spectral details and rapid variations

## 💪 Model Training in Google Colab
We collect and organize audio samples for each class, extract MFCC features, and use them to train a lightweight neural network.

The training pipeline includes:

- Loading the dataset  
- MFCC Feature Extraction, Normalisation and Dataset split
- Training a **1-D Convolutional Neural Network (CNN)**  
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

<!-- The implementation is optimized to meet **memory and timing constraints** of embedded systems. -->


## ⚙️ Instructions

- The data was collected on the Arduino with [ArduinoSamples folder](https://github.com/ilkabret/LowPowerES/tree/main/Assignment_2/ArduinoSamples)
- All .txt files are stored in the [dataset folder](https://github.com/ilkabret/LowPowerES/tree/main/Assignment_2/dataset).  
- Open the Colab notebook `KWS_MFCC_Training.ipynb`, upload the .txt-files into a folder called "dataset", and run all cells to train the model.  
- The last cell will provide "kws_model_data.h" and "kws_params.h"
- Upload the Arduino sketch TinyML_Application.ino (including /kws_model_data.h, /kws_params.h and /mfcc_tables.h) from the [KWS_Arduino folder](https://github.com/ilkabret/LowPowerES/tree/main/Assignment_2/KWS_Arduino) to the board.
- Open the Serial Monitor and test real-time keyword recognition by either clapping, tapping, snapping or whistling.  

---

Ilka BRETSCHNEIDER (268664)<br> Romain NOBLET (268709)

*Università degli Studi di Trento - 2026*