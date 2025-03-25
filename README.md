# 😷 Mask Detection with MobileNetV2

A deep learning-based face mask detection system that classifies whether a person is wearing a mask using a MobileNetV2 architecture. The system is built using TensorFlow and Keras.

This project follows the tutorial from [YouTube](https://www.youtube.com/watch?v=TNZAbVNTLhA&t=7701s).

---

## Project Structure

## Project Structure

- **dataset/**: Folder containing the dataset for training
- **examples/**: Folder with sample images for testing
- **mobilenet_v2.model**: Saved model file after training
- **mask_detection.ipynb**: Jupyter notebook for training the model
- **use_model.ipynb**: Jupyter notebook for static image inference
- **use_model_video.ipynb**: Jupyter notebook for real-time webcam detection
- **deploy.prototxt**: Prototxt file for the face detection model
- **res10_300x300_ssd_iter_140000.caffemodel**: Pre-trained Caffe model for face detection

---

## Overview

This project aims to detect whether people are wearing a mask or not using the following steps:

### 1. Dataset Preparation
The dataset consists of images organized into two folders: `with_mask` and `without_mask`. These images are loaded and preprocessed to be compatible with the MobileNetV2 architecture. The images are resized to 224x224 pixels and normalized using `preprocess_input`.

### 2. Model Training
The model is based on **MobileNetV2**, which is used as the backbone. A custom head model is added to classify the images as either `with_mask` or `without_mask`. The model is trained using the dataset, and **data augmentation** is applied to increase the robustness of the model.

### 3. Model Evaluation
After training, the model is evaluated using the test set and generates performance metrics such as accuracy, precision, recall, and F1 score.

### 4. Mask Detection - Static Image
The trained model is used to predict whether people in static images are wearing a mask. You can run the Jupyter notebook `use_model.ipynb` for inference on individual images.

### 5. Mask Detection - Real-Time Webcam
Using the OpenCV library, real-time face detection and mask prediction are implemented. The `use_model_video.ipynb` file is used to detect faces in the webcam feed and classify whether the person is wearing a mask or not.

---

## Requirements

To run the project, install the following dependencies:

```bash
pip install imutils opencv-python tensorflow scikit-learn matplotlib
