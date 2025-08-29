"""
Model factory for creating CNN models.

This module provides a unified interface for creating different CNN architectures
for satellite detection and segmentation tasks.
"""

import logging
from typing import Dict, Any, Optional
import torch.nn as nn

from .unet import create_unet_model
from .fasterrcnn import create_fasterrcnn_model
from .yolov8 import create_yolov8_model

logger = logging.getLogger(__name__)


# Model registry
MODEL_REGISTRY = {
    'unet': create_unet_model,
    'fasterrcnn': create_fasterrcnn_model,
    'yolov8': create_yolov8_model,
}


class ModelFactory:
    """Factory class for creating CNN models."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model based on type and configuration.
        
        Args:
            model_type: Type of model to create ('unet', 'fasterrcnn', 'yolov8')
            config: Model configuration dictionary
            
        Returns:
            Initialized model
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in MODEL_REGISTRY:
            available_models = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available models: {available_models}"
            )
        
        logger.info("Creating %s model", model_type)
        
        # Get the appropriate factory function
        factory_fn = MODEL_REGISTRY[model_type]
        
        # Create and return the model
        model = factory_fn(config)
        
        logger.info(
            "Successfully created %s model with %d parameters",
            model_type, 
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        
        return model
    
    @staticmethod
    def list_available_models() -> list:
        """
        Get list of available model types.
        
        Returns:
            List of available model type strings
        """
        return list(MODEL_REGISTRY.keys())
    
    @staticmethod
    def register_model(name: str, factory_fn: callable):
        """
        Register a new model type.
        
        Args:
            name: Name of the model type
            factory_fn: Factory function that creates the model
        """
        MODEL_REGISTRY[name] = factory_fn
        logger.info("Registered new model type: %s", name)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model directly from configuration.
    
    Args:
        config: Configuration dictionary containing model specification
        
    Returns:
        Initialized model
        
    Example:
        config = {
            'model': {
                'type': 'unet',
                'unet': {
                    'n_channels': 3,
                    'n_classes': 2,
                    'depth': 4
                }
            }
        }
        model = create_model_from_config(config)
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'unet')
    
    return ModelFactory.create_model(model_type, config)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_class': model.__class__.__name__,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    return info


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'model' not in config:
        raise ValueError("Configuration must contain 'model' section")
    
    model_config = config['model']
    
    if 'type' not in model_config:
        raise ValueError("Model configuration must specify 'type'")
    
    model_type = model_config['type']
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Invalid model type: {model_type}. Available: {available}")
    
    # Type-specific validation
    if model_type == 'unet':
        unet_config = model_config.get('unet', {})
        required_keys = ['n_channels', 'num_classes']
        for key in required_keys:
            if key not in unet_config:
                logger.warning("UNet config missing optional key: %s", key)
    
    elif model_type == 'fasterrcnn':
        fasterrcnn_config = model_config.get('fasterrcnn', {})
        if 'num_classes' in fasterrcnn_config and fasterrcnn_config['num_classes'] < 2:
            raise ValueError("Faster R-CNN must have at least 2 classes (including background)")
    
    elif model_type == 'yolov8':
        yolo_config = model_config.get('yolov8', {})
        if 'num_classes' in yolo_config and yolo_config['num_classes'] < 1:
            raise ValueError("YOLOv8 must have at least 1 class")
    
    logger.info("Model configuration validation passed")
    return True


# Convenience functions for common configurations
def create_unet_satellite_model(
    n_channels: int = 3,
    num_classes: int = 1,  # Detection classes (satellites)
    depth: int = 4,
    use_attention: bool = False,
    dropout_rate: float = 0.1
) -> nn.Module:
    """
    Create U-Net model for satellite detection with common defaults.
    
    Args:
        n_channels: Number of input channels
        num_classes: Number of detection classes
        depth: Network depth
        use_attention: Use attention gates
        dropout_rate: Dropout rate
        
    Returns:
        Configured U-Net model for detection
    """
    config = {
        'model': {
            'type': 'unet',
            'unet': {
                'n_channels': n_channels,
                'num_classes': num_classes,
                'depth': depth,
                'use_attention': use_attention,
                'dropout_rate': dropout_rate
            }
        }
    }
    
    return create_model_from_config(config)


def create_fasterrcnn_satellite_model(
    num_classes: int = 2,
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = True,
    min_size: int = 800,
    max_size: int = 1333
) -> nn.Module:
    """
    Create Faster R-CNN model for satellite detection with common defaults.
    
    Args:
        num_classes: Number of classes (including background)
        backbone_name: Backbone architecture
        pretrained_backbone: Use pretrained weights
        min_size: Minimum input size
        max_size: Maximum input size
        
    Returns:
        Configured Faster R-CNN model
    """
    config = {
        'model': {
            'type': 'fasterrcnn',
            'fasterrcnn': {
                'num_classes': num_classes,
                'backbone_name': backbone_name,
                'pretrained_backbone': pretrained_backbone,
                'min_size': min_size,
                'max_size': max_size
            }
        }
    }
    
    return create_model_from_config(config)


def create_yolov8_satellite_model(
    num_classes: int = 1,  # Detection classes (no background for YOLO)
    input_channels: int = 3,
    width_mult: float = 1.0,
    depth_mult: float = 1.0
) -> nn.Module:
    """
    Create YOLOv8 model for satellite detection with common defaults.
    
    Args:
        num_classes: Number of detection classes
        input_channels: Number of input channels
        width_mult: Width multiplier for model scaling
        depth_mult: Depth multiplier for model scaling
        
    Returns:
        Configured YOLOv8 model
    """
    config = {
        'model': {
            'type': 'yolov8',
            'yolov8': {
                'num_classes': num_classes,
                'input_channels': input_channels,
                'width_mult': width_mult,
                'depth_mult': depth_mult
            }
        }
    }
    
    return create_model_from_config(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test model factory
    print("Available models:", ModelFactory.list_available_models())
    
    # Test creating models with different configurations
    configs = [
        {
            'model': {
                'type': 'unet',
                'unet': {
                    'n_channels': 3,
                    'num_classes': 1,
                    'depth': 4
                }
            }
        },
        {
            'model': {
                'type': 'fasterrcnn',
                'fasterrcnn': {
                    'num_classes': 2,
                    'backbone_name': 'resnet50'
                }
            }
        },
        {
            'model': {
                'type': 'yolov8',
                'yolov8': {
                    'num_classes': 1,
                    'input_channels': 3
                }
            }
        }
    ]
    
    for i, config in enumerate(configs):
        try:
            validate_model_config(config)
            model = create_model_from_config(config)
            info = get_model_info(model)
            print(f"\nModel {i+1}: {config['model']['type']}")
            print(f"  Parameters: {info['trainable_parameters']:,}")
            print(f"  Size: {info['model_size_mb']:.1f} MB")
        except Exception as e:
            print(f"Error creating model {i+1}: {e}")
    
    # Test convenience functions
    print("\nTesting convenience functions:")
    
    try:
        unet = create_unet_satellite_model(num_classes=1, use_attention=True)
        print(f"U-Net: {get_model_info(unet)['trainable_parameters']:,} parameters")
    except Exception as e:
        print(f"Error creating U-Net: {e}")
    
    try:
        fasterrcnn = create_fasterrcnn_satellite_model(num_classes=2)
        print(f"Faster R-CNN: {get_model_info(fasterrcnn)['trainable_parameters']:,} parameters")
    except Exception as e:
        print(f"Error creating Faster R-CNN: {e}")
    
    try:
        yolo = create_yolov8_satellite_model(num_classes=1)
        print(f"YOLOv8: {get_model_info(yolo)['trainable_parameters']:,} parameters")
    except Exception as e:
        print(f"Error creating YOLOv8: {e}")
