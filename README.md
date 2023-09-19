# SigRec
# TensorFlow Signature Recognition Code Documentation

This documentation explains signature recognition using the TensorFlow framework. The code employs a convolutional neural network (CNN) to classify signatures as genuine or forged.

Data Used:
https://cedar.buffalo.edu/NIJ/data/

Model Trained:
https://drive.google.com/file/d/1Qauvbk6vq-9oHqKAMfX1JgcOC6_uoVdm/view?usp=sharing

# TensorFlow Signature Recognition Code Documentation

This documentation explains a Python script for signature recognition using the TensorFlow framework. The code employs a convolutional neural network (CNN) to classify signatures as genuine or forged.

## Table of Contents
1. **Introduction**
2. **Dependencies**
3. **GPU Configuration**
4. **Data Preprocessing**
5. **Model Setup**
6. **Model Training**
7. **Model Testing**
8. **Model Saving**

### 1. Introduction
This code is designed to perform signature recognition using a machine learning model. The model is trained on a dataset of genuine and forged signatures and later tested for accuracy.

### 2. Dependencies
- TensorFlow: The deep learning framework used for building and training the model.
- NumPy: A library for numerical operations and array manipulations.
- Matplotlib: A visualization library for creating plots and charts.

The code imports necessary modules and libraries at the beginning.

### 3. GPU Configuration
This section configures GPU memory growth for TensorFlow. It ensures that GPU memory is allocated on an as-needed basis, which can help prevent memory overflow.

### 4. Data Preprocessing
The dataset is loaded using `tf.keras.utils.image_dataset_from_directory`. It assumes that the dataset is organized in directories with two subdirectories, one for genuine signatures and one for forged signatures. The dataset is split into training, validation, and test sets.

### 5. Model Setup
A convolutional neural network (CNN) model is set up using TensorFlow's Keras API. It consists of convolutional layers, max-pooling layers, flattening, and fully connected layers. The model is compiled with the Adam optimizer and binary cross-entropy loss.

### 6. Model Training
The model is trained on the training dataset using `model.fit`. Training data is specified by the `train` dataset, and validation data is specified by the `val` dataset. Training is performed for a specified number of epochs.

### 7. Model Testing
The trained model is evaluated on a test dataset. Precision, recall, and binary accuracy metrics are calculated using TensorFlow's built-in metrics classes (`Precision`, `Recall`, `BinaryAccuracy`). These metrics are updated for each batch in the test dataset, and the results are printed.

### 8. Model Saving
The trained model is saved to a file in the "models" directory using `model.save`. This saved model can be loaded later for inference or further training.

## Note
- The code assumes that you have a dataset organized in the "data" directory with subdirectories for genuine and forged signatures.
- The model architecture and hyperparameters (e.g., number of layers, epochs) can be adjusted to improve performance.
- This is a simplified example for educational purposes. In a real-world application, more data preprocessing, data augmentation, and hyperparameter tuning may be necessary for better results.
