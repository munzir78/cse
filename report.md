# Project Report: Skin Type Detection Using Deep Learning

## Abstract

Skin type detection is a crucial aspect of dermatology and skincare. It helps individuals choose suitable skincare products and aids dermatologists in diagnosing skin conditions. This project focuses on building an AI-based model to classify skin types using deep learning techniques. We employed ResNet50, EfficientNet, and a hybrid model combining the strengths of both architectures. The models were trained on a carefully curated dataset and evaluated for accuracy, with ResNet50 achieving 92%, EfficientNet achieving 93%, and the hybrid model achieving the highest accuracy of 94%.

## Introduction

Skin types can be classified into different categories, such as normal, dry, oily, combination, and sensitive skin. Traditional skin type classification relies on expert dermatologists and subjective assessment, making it prone to human error. In this project, we developed a deep learning model that automates skin type classification using image processing techniques, convolutional neural networks (CNNs), and transfer learning.

## Dataset Collection and Preprocessing

A high-quality dataset is essential for training a robust machine-learning model. We gathered skin images from publicly available dermatology datasets, supplemented with additional images from various sources. The dataset was balanced across all skin types to prevent bias in model predictions.

### Image Acquisition:

- Collected images from dermatology datasets and web sources.
- Ensured data diversity by considering images from different ethnicities, lighting conditions, and resolutions.

### Data Cleaning and Augmentation:

- Resized all images to a uniform dimension of 224x224 pixels.
- Applied augmentation techniques such as rotation, flipping, brightness adjustments, and noise addition to enhance model generalization.
- Normalized pixel values to scale them between 0 and 1.
- Split the dataset into 80% training, 10% validation, and 10% test sets.

## Model Selection and Implementation

We explored different CNN architectures to identify the best-performing model for skin type classification.

### ResNet50 Model

- ResNet50, a 50-layer deep convolutional neural network, was chosen due to its ability to mitigate the vanishing gradient problem using residual connections.
- Used pre-trained weights from ImageNet to leverage transfer learning.
- Replaced the final dense layer with a fully connected layer having five output nodes (one for each skin type) and a softmax activation function.
- Applied batch normalization and dropout to prevent overfitting.
- Optimized using Adam optimizer with a learning rate of 0.0001.
- Achieved an accuracy of 92%.

### EfficientNet Model

- EfficientNet is a family of CNN models known for achieving high accuracy with fewer parameters.
- Implemented EfficientNet-B0 for its efficiency in training.
- Fine-tuned the model using pre-trained ImageNet weights.
- Utilized depth-wise separable convolutions to reduce computational complexity.
- Achieved an accuracy of 93%.

### Hybrid Model (ResNet50 + EfficientNet)

- To further enhance performance, we developed a hybrid model by combining features from ResNet50 and EfficientNet.
- Extracted feature maps from both models and concatenated them before passing them to a fully connected layer.
- Implemented weighted ensembling to give more importance to EfficientNet predictions while preserving the robustness of ResNet50.
- Applied dropout and batch normalization layers to stabilize training.
- Achieved the highest accuracy of 94%.

## Model Training and Evaluation

The models were trained using the TensorFlow and Keras frameworks on a high-performance GPU setup. The training pipeline included:

- Using categorical cross-entropy as the loss function.
- Training each model for 50 epochs with early stopping to prevent overfitting.
- Monitoring validation loss and accuracy to fine-tune hyperparameters.

### Performance Metrics:

- **ResNet50**: Accuracy - 92%, Precision - 91%, Recall - 90%.
- **EfficientNet**: Accuracy - 93%, Precision - 92%, Recall - 91%.
- **Hybrid Model**: Accuracy - 94%, Precision - 93%, Recall - 92%.

## Conclusion

This project successfully developed an AI-based skin type detection system using deep learning. The hybrid model achieved the highest accuracy of 94%, proving the effectiveness of combining multiple architectures. The deployed API enables easy integration into skincare applications, benefiting dermatologists and consumers alike. Future improvements may include expanding the dataset and incorporating additional features such as age, gender, and skin conditions for more personalized recommendations.
