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

## 🏗️ Project Structure

```
satellite-detection-segmentation/
├── data/                       # Data storage and management
│   ├── raw/                   # Original datasets
│   ├── processed/             # Preprocessed data
│   └── splits/                # Train/val/test splits
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model implementations
│   │   ├── cnn/              # CNN architectures
│   │   └── transformers/     # Transformer models
│   ├── training/             # Training pipeline
│   ├── evaluation/           # Evaluation and metrics
│   ├── inference/            # Inference pipeline
│   └── utils/                # Utilities and helpers
├── configs/                   # Configuration files
├── notebooks/                 # Jupyter notebooks for analysis
├── docker/                    # Docker configurations
├── scripts/                   # Training and evaluation scripts
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Keras 3.0+
- HuggingFace Transformers
- MLflow
- Docker

### Installation
```bash
git clone https://github.com/el97179/satellite-detection-segmentation.git
cd satellite-detection-segmentation
pip install -r requirements.txt
```

## 📊 Dataset

The project uses:
- **Training Data**: 10,000 synthetic satellite images from NAPA2_Audacity_v2_training
- **Ground Truth**: Bounding boxes and segmentation masks in CSV format
- **Test Data**: Real satellite images from spacecloud/ folder

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

## 🐳 Docker Support

Containerized environments for:
- **Training**: GPU-enabled training container
- **Inference**: Lightweight inference service
- **Development**: Full development environment

## 📝 Contributing

Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Project Documentation](docs/)
- [Model Architecture Details](docs/models.md)
- [Training Guide](docs/training.md)
- [API Documentation](docs/api.md)
