# Automatic Fire Detection Using CNN and Computer Vision

## Overview

This project presents an AI-based fire detection system designed for use in server rooms or other sensitive environments. It utilizes a Convolutional Neural Network (CNN) built on MobileNetV2 to identify the presence of fire from camera input. The system can be integrated with an autonomous drone to perform real-time surveillance and trigger a response action (e.g., activating an actuator to extinguish a fire).

The core objective is to detect visible flames in images or live camera footage with high accuracy and reliability, and to simulate the actuator's response for integration into a drone's onboard system.

---

## Project Structure

```

Automatic\_fire\_detection\_CNN/
├── CNN\_training.ipynb               # Model training and saving
├── flame\_detection\_camera.ipynb     # Real-time detection using a camera
├── flame\_detection\_image.ipynb      # Static image detection
├── flame\_detector.pth               # Trained model file (can be linked externally)
└── README.md                        # Project documentation

```

---

## Features

- Trains a lightweight MobileNetV2 CNN on a labeled fire/non-fire dataset.
- Performs real-time fire detection using a webcam or drone-mounted camera.
- Supports classification of manually uploaded image files.
- Includes simulated actuator trigger (`print("Fire detected! sending signal to actuator")`) for later replacement with hardware integration.
- Designed for integration into autonomous systems, such as inspection drones.

---

## Dataset

The project uses the **Fire Dataset by Phylake1337** available on Kaggle:

**Dataset Link:**  
https://www.kaggle.com/datasets/phylake1337/fire-dataset

**Expected folder structure after download:**

```

fire\_dataset/
├── fire\_images/
│   ├── fire.1.png
│   ├── fire.2.png
│   └── ...
├── non\_fire\_images/
│   ├── non\_fire.1.png
│   ├── non\_fire.2.png
│   └── ...

````

This dataset contains labeled images of fire and non-fire scenes suitable for binary classification.

---

## Model Architecture

The model is based on **MobileNetV2**, a lightweight CNN optimized for real-time performance on low-power hardware.

- **Backbone:** MobileNetV2 (from torchvision models)
- **Classifier:** Modified final layer to output 2 classes (fire, non-fire)
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam
- **Accuracy Achieved:** ~99% on validation data after training

The model is saved in PyTorch format as `flame_detector.pth`.

---

## Notebooks

### 1. `CNN_training.ipynb`
- Loads and preprocesses the dataset
- Applies transformations for normalization and resizing
- Splits data into training and validation sets
- Trains the CNN and saves the model

### 2. `flame_detection_camera.ipynb`
- Loads the trained model
- Captures real-time frames from a connected camera (e.g., drone-mounted camera or webcam)
- Processes each frame and classifies as fire or non-fire
- Prints a simulated actuator trigger if fire is detected

### 3. `flame_detection_image.ipynb`
- Loads the trained model
- Accepts a manually specified image path
- Applies preprocessing and classifies the image
- Outputs the prediction (fire or non-fire)

---


## Clone the Repository

```bash
git clone https://github.com/ahmdmohamedd/Automatic_fire_detection_CNN.git
cd Automatic_fire_detection_CNN
````


### 2. Download the Dataset

Download the dataset from Kaggle and extract it to the root directory as `fire_dataset/`.

---

## Usage

### A. Training the Model

Run `CNN_training.ipynb` in Jupyter to train and save the model.

### B. Real-Time Camera Detection

Run `flame_detection_camera.ipynb` to use live camera feed for flame detection. If fire is detected, the system will print a message indicating that a signal is being sent to the actuator.

> Note: Ensure your camera is accessible via OpenCV. On headless servers or environments without GUI support, `cv2.imshow()` may not work and can be commented out.

### C. Static Image Detection

Run `flame_detection_image.ipynb`, update the `test_path` variable with the path to an image file, and run the notebook. It will print whether fire was detected.

---

## Model Deployment Notes

* The model is lightweight and can be deployed on edge devices such as Jetson Nano or Raspberry Pi with camera modules.
* For real-world integration, the `print` statement can be replaced with hardware-level signals (e.g., GPIO, serial commands).
* Further safety and robustness checks should be implemented for industrial environments.

---

## Limitations and Future Work

* The model is trained on visible flames and may not detect early-stage or obscured fires.
* Performance under varying lighting conditions and angles should be validated.
* Integration with thermal imaging or smoke detection could improve reliability.
* Currently, only flame detection is implemented; heat, smoke, or other fire indicators are not included.

---
