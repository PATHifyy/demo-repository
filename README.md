# PATHify: Plant Health Identifier

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow to classify plant diseases into 38 categories. The model is designed to assist in identifying plant diseases based on image data, with the goal of supporting sustainable agriculture and improving crop management practices.

## Features
- **Input Shape:** The model processes input images of size `(128, 128, 3)`.
- **Architecture:**
  - 7 Convolutional Layers (`Conv2D`) for feature extraction.
  - 4 Max Pooling Layers (`MaxPool2D`) for spatial downsampling.
  - 2 Fully Connected Layers (`Dense`) for classification, including a final `softmax` layer for multi-class output.
- **Performance:** Achieves an accuracy range of 85-95% on the classification task.

## Dependencies
The following Python libraries are required to run the notebook:
- TensorFlow
- NumPy
- Matplotlib (optional, for visualizing results)

Ensure you have these installed using:
```bash
pip install tensorflow numpy matplotlib
```

## Model Architecture
The CNN model is structured as follows:
1. **Convolutional Blocks:**
   - Block 1: Two `Conv2D` layers with 32 filters and ReLU activation, followed by a `MaxPool2D` layer.
   - Block 2: Two `Conv2D` layers with 64 filters and ReLU activation, followed by a `MaxPool2D` layer.
   - Block 3: Two `Conv2D` layers with 128 filters and ReLU activation, followed by a `MaxPool2D` layer.
   - Block 4: One `Conv2D` layer with 256 filters and ReLU activation, followed by a `MaxPool2D` layer.
2. **Fully Connected Layers:**
   - Flattened output from convolutional blocks.
   - Dense layer with 512 units and ReLU activation.
   - Dropout layer to reduce overfitting.
   - Final `Dense` layer with 38 units and softmax activation for classification.

## Usage
1. **Prepare the Dataset:**
   - Ensure your dataset is preprocessed and split into training, validation, and test sets.
   - Images should be resized to `(128, 128, 3)`.

2. **Run the Notebook:**
   - Load the dataset and preprocess it.
   - Execute the notebook cells to train and evaluate the CNN model.

3. **Evaluate the Model:**
   - Check the model's performance metrics, such as accuracy and loss, on the validation and test sets.

## Key Results
- Achieved an accuracy of 90-95% on the dataset.
- Demonstrated the capability to classify plant diseases effectively, supporting early detection and intervention.

## Future Improvements
- Incorporate data augmentation to enhance the dataset.
- Explore transfer learning using pre-trained models for improved performance.
- Optimize hyperparameters for better accuracy and generalization.
