# Gesture Recognition Application Using Feature Extraction

## 📌 Project Overview
This project implements a **real-time gesture recognition system** using motion sensor data from an **Arduino-compatible IMU** (accelerometer and/or gyroscope). The system uses **feature extraction techniques** and performs **inference directly on the device**.

The full pipeline includes **data collection**, **feature extraction**, **model training in Google Colab**, and **deployment of the trained model on Arduino for on-device inference**.

### 📊 Data collection

#### ✋ Supported gestures

- Punch
- Flex
- RightLeft
- Circle

#### Feature extraction - 6 Features

##### 1. Mean
The mean represents the average value of the signal within the window.

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

Where:
- $\mu$ = mean value of the signal
- $x_i$ = the i-th sample of the signal in the window
- $N$ = number of samples in the window (119 in this implementation)

##### 2. Standard Deviation
Standard deviation measures how much the signal varies around the mean.

$$
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}
$$

Where:
- $\sigma$ = standard deviation of the signal
- $x_i$ = the i-th sample of the signal in the window
- $\mu$ = mean value of the signal
- $N$ = number of samples in the window

##### 3. Root Mean Square
RMS represents the overall magnitude of the signal.

$$
RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2}
$$

Where:
- $RMS$ = root mean square value of the signal
- $x_i$ = the i-th sample of the signal in the window
- $N$ = number of samples in the window

##### 4. Minimum Value
The minimum feature captures the lowest signal value within the window.

$$
x_{min} = \min(x_1, x_2, ..., x_N)
$$

Where:
- $x_{min}$ = minimum signal value in the window
- $x_1, x_2, ..., x_N$ = signal samples within the window
- $N$ = number of samples in the window

##### 5. Maximum Value
The maximum feature captures the largest signal value within the window.

$$
x_{max} = \max(x_1, x_2, ..., x_N)
$$

Where:
- $x_{max}$ = maximum signal value in the window
- $x_1, x_2, ..., x_N$ = signal samples within the window
- $N$ = number of samples in the window

##### 6. Signal Energy (Power Spectral Density Approximation)
This feature estimates the energy (power) of the signal within the window.

$$
E = \frac{1}{N}\sum_{i=1}^{N} x_i^2
$$

Where:
- $E$ = signal energy estimate
- $x_i$ = the i-th sample of the signal in the window
- $N$ = number of samples in the window

### 💪 Model Training in Google Colab
We load the data for four gestures, extract the features, normalize them, and use them as inputs for our model with one-hot encoded labels.
This is then used to train a small neural network in TensorFlow using the Adam optimizer for 200 epochs with a batch size of 8, validating the model on a separate dataset.

### 💻 Deployment of the Trained Model on Arduino for on-device Inference
We deploy the trained TensorFlow Lite model to Arduino by converting it into a C header file (model.h) and allocating a small tensor arena so the board can run inference locally.
During execution, the Arduino collects data, extracts the same 36 normalized features used in training, feeds them into the model, and then prints the gesture with the highest predicted probability.

## ⚙️ Intructions

- All CSV files are stored in the /data folder.
- Open the Colab notebook /training.ipynb, upload the .csv files and execute the script to to train the model. The trained model will provide a .tflite file and a model.h file.
- Upload the Arduino sketch /TinyML_Application.ino (including /model.h) to the board.
- Try some gestures and check the output on the Serial Monitor.

---

Ilka BRETSCHNEIDER (268664)<br> Romain NOBLET (268709)

*Università degli Studi di Trento - 2026*
