# Gesture Recognition Application Using Feature Extraction

## 🎯 Project Overview
This project implements a **real-time gesture recognition system** using motion sensor data from an **Arduino-compatible IMU** (accelerometer and/or gyroscope). The system uses feature extraction techniques such as **RMS** and **PSD** to classify multiple gesture types and performs **inference directly on the device**.

The full pipeline includes **data collection**, **feature extraction**, **model training in Google Colab**, and **deployment of the trained model on Arduino for on-device inference**.

### Intructions

- All CSV files are stored in the /data folder. Make sure they are accessible in your Colab notebook.
- Open and run the Colab notebook to train the model. The trained model will be converted to a .tflite file and exported as model.h.
- Upload the Arduino sketch (including model.h) to your board.

**Usage**: Open the Serial Monitor to see live gesture predictions.

<br>

> Ilka BRETSCHNEIDER <br> Romain NOBLET

*Università degli Studi di Trento - 2026*
