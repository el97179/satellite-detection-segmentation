# Satellite Detection and Segmentation

A comprehensive machine learning project for satellite detection and segmentation using both CNN-based and Transformer-based approaches.

## 🎯 Project Overview

This project implements multiple state-of-the-art models for satellite detection and segmentation:

### CNN-Based Models
- **U-Net with Detection Head**: Excellent for segmentation with skip connections
- **Mask R-CNN**: State-of-the-art instance segmentation 
- **YOLOv8-Segment**: Fast single-stage detection and segmentation

### Transformer-Based Models  
- **DETR**: End-to-end object detection without NMS
- **Mask2Former**: Unified architecture for all segmentation tasks
- **SegFormer**: Efficient transformer for segmentation

## 📊 Dataset

The project uses:
- **Training Data**: 10,000 synthetic satellite images from NAPA2_Audacity_v2_training
- **Ground Truth**: Bounding boxes and segmentation masks in CSV format
- **Test Data**: Real satellite images from spacecloud/ folder

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/el97179/satellite-detection-segmentation.git
cd satellite-detection-segmentation
pip install -r requirements.txt
```

## 🔧 Framework Support

- **PyTorch**: Primary framework for CNN and transformer models
- **Keras 3**: Multi-backend support (JAX, TensorFlow, PyTorch)
- **HuggingFace**: Pre-trained transformer models

## 📈 Experiment Tracking

All experiments are tracked using MLflow with:
- Model parameters and hyperparameters
- Training and validation metrics
- Model artifacts and checkpoints
- Performance visualizations
