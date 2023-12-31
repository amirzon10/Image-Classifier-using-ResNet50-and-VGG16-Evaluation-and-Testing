# ImageClassifier_ResNet50_VGG16

## Project Objective

This project was a requirement to complete the IBM AI Engineering Professional Certificate. Its primary aim was to demonstrate proficiency in utilizing convolutional neural network models like ResNet50 and VGG16 for image classification tasks.

## Description of Models

ResNet50 and VGG16 are convolutional neural network (CNN) models extensively used in computer vision tasks like image classification, object detection, and segmentation.

### VGG16
VGG16, introduced by the Visual Geometry Group at the University of Oxford in 2014, comprises 16 layers of convolutional and fully connected layers. Known for its simplicity and uniformity, VGG16's architecture features 3x3 filter sizes and a stride of 1 pixel in all its convolutional layers. It has found applications in image recognition, object detection, and segmentation.

### ResNet50
Introduced in 2015 by Microsoft Research Asia, ResNet50 is a variant of the ResNet architecture addressing the vanishing gradients issue in deep neural networks using skip connections. With 50 layers, ResNet50 has demonstrated state-of-the-art performance in various image classification benchmarks, including ImageNet.

## Evaluation Metrics
The evaluation of the models relied on accuracy and loss metrics.

## Dataset Information

The dataset utilized for this project was sourced from [Concrete Crack Images for Classification](http://dx.doi.org/10.17632/5y9wdsg2zt.2#file-c0d86f9f-852e-4d00-). It presumably contains images relevant to concrete crack classification.

## Required Libraries

The project utilized Keras library with specific imports:
```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
