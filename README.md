# Expiration date recognition

## Overview
This project aims to develop a system for recognizing expiry dates on products in supermarkets. The system utilizes two main models: a detector model for detecting and localizing expiry date regions in images, and an image-to-text model for extracting the actual date information from the detected regions.

The project is implemented in Python using the PyTorch library for deep learning tasks. Additionally, it leverages a dataset provided by Hugging Face, a platform for sharing and accessing natural language processing (NLP) datasets and models.

## Components
### Detector Model
The detector model is responsible for identifying regions within an image that contain expiry dates. It is based on a deep learning architecture, specifically a Faster R-CNN (Region-based Convolutional Neural Network), which is trained to detect objects within images. In this case, the model is trained to detect and localize expiry date regions.

To check how it works go to notebooks/detect.ipnyb

### Image-to-Text Model
TBD