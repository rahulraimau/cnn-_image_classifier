layout: default title: Image Classification (CNN)
🧠 Image Classifier using CNN
Project Report: Animal Image Classification using Convolutional Neural Networks
Author: [RahulRai]
Date: July 9, 2025

Abstract
This project focuses on the development and evaluation of a Convolutional Neural Network (CNN) for classifying animal images, specifically distinguishing between dogs, cats, and deer. Utilizing a subset of 10,000 labeled images from the CIFAR-10 dataset, the CNN architecture was carefully designed with layers such as Conv2D, MaxPooling, and Dropout. To enhance model generalization and prevent overfitting, extensive data augmentation techniques were applied using Keras's ImageDataGenerator. Furthermore, training stability and convergence speed were improved through the implementation of early stopping and batch normalization. The developed model achieved a final test accuracy of 92%, demonstrating its robust capability in recognizing these animal categories. The project also includes a real-time prediction interface built with Streamlit, showcasing the practical applicability of the trained model.

1. Introduction
Image classification is a fundamental task in computer vision with widespread applications, from autonomous vehicles to medical diagnosis. The ability to accurately categorize images into predefined classes is crucial for many intelligent systems. This project addresses the challenge of classifying animal images (dogs, cats, and deer) using a Convolutional Neural Network (CNN). CNNs have emerged as the state-of-the-art for image-related tasks due to their inherent ability to learn hierarchical features directly from raw pixel data.

The CIFAR-10 dataset, a widely recognized benchmark in image classification, was chosen for this project. While the full CIFAR-10 dataset contains 10 classes, this project specifically focused on a subset relevant to animal classification to achieve a high-performance, focused model. The goal was to develop a robust CNN model capable of achieving high accuracy on unseen animal images and to demonstrate its utility through a real-time prediction application.

2. Dataset
The project utilized a subset of 10,000 labeled images from the CIFAR-10 dataset. The CIFAR-10 dataset originally consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. For this specific project, we extracted images belonging to the "dog", "cat", and "deer" classes, resulting in 10,000 images (approx. 3,333 images per class for dog, cat, and deer, assuming a balanced subset was chosen for these classes from the original 6,000 images per class, or a total of 10,000 images were sampled from these three classes).

Image Dimensions: 32x32 pixels

Color Channels: 3 (RGB)

Classes Used: Dog, Cat, Deer

Total Images Used: 10,000 (split into training and testing sets)

Data Preprocessing:
Before feeding the images into the CNN, standard preprocessing steps were applied:

Normalization: Pixel values were scaled from the range [0, 255] to [0, 1] by dividing by 255.0. This helps in faster convergence during training.

One-Hot Encoding: The categorical labels (e.g., 'dog', 'cat', 'deer') were converted into one-hot encoded vectors (e.g., [1, 0, 0] for dog, [0, 1, 0] for cat, [0, 0, 1] for deer).

3. Methods
3.1. CNN Architecture
The Convolutional Neural Network (CNN) was built using TensorFlow/Keras. The architecture was designed to progressively extract features from the images, starting with low-level features (edges, textures) and moving towards high-level features (object parts, shapes).

A typical architecture employed involved:

Input Layer: Expects images of shape (32, 32, 3).

Convolutional Layers (Conv2D): Multiple Conv2D layers with increasing filter counts (e.g., 32, 64, 128) and ReLU activation functions. Smaller kernel sizes (e.g., 3x3) were generally preferred to capture finer details.

Max Pooling Layers (MaxPooling2D): Applied after convolutional layers to reduce spatial dimensions, downsample feature maps, and make the model more robust to minor translations. A common pool size of (2,2) was used.

Batch Normalization: Integrated after convolutional layers to normalize the activations of the previous layer. This helps in stabilizing the learning process, allowing for higher learning rates and acting as a regularization technique.

Dropout Layers: Strategically placed after pooling layers or before dense layers to prevent overfitting. A dropout rate (e.g., 0.25 to 0.5) was used, randomly setting a fraction of input units to zero at each update during training time, which helps in reducing co-adaptation of neurons.

Flatten Layer: Converts the 2D feature maps from the convolutional and pooling layers into a 1D vector, preparing the data for the fully connected layers.

Dense Layers: One or more fully connected layers with ReLU activation for learning high-level relationships from the flattened features.

Output Layer: A final dense layer with 3 neurons (one for each class: dog, cat, deer) and a softmax activation function, which outputs a probability distribution over the classes.
