"""
Test script for CNN model implementations.

This script tests all CNN models with different configurations to ensure
they work correctly with the satellite detection and segmentation dataset.
"""

import sys
import os
import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import (
    ModelFactory, 
    create_model_from_config,
    get_model_info,
    validate_model_config,
    create_unet_satellite_model,
    create_maskrcnn_satellite_model,
    create_yolov8_satellite_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/base_config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", config_path)
        return config
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
        return {}


def test_model_creation(model_type: str, config: dict) -> bool:
    """
    Test model creation for a specific type.
    
    Args:
        model_type: Type of model to test
        config: Configuration dictionary
        
    Returns:
        True if test passed, False otherwise
    """
    try:
        logger.info("Testing %s model creation...", model_type)
        
        # Update config for this model type
        test_config = config.copy()
        test_config['model']['type'] = model_type
        
        # Validate configuration
        validate_model_config(test_config)
        
        # Create model
        model = create_model_from_config(test_config)
        
        # Get model info
        info = get_model_info(model)
        
        logger.info("✓ %s model created successfully", model_type)
        logger.info("  - Parameters: %s", f"{info['trainable_parameters']:,}")
        logger.info("  - Size: %.1f MB", info['model_size_mb'])
        
        return True
        
    except Exception as e:
        logger.error("✗ Failed to create %s model: %s", model_type, e)
        return False


def test_model_forward_pass(model_type: str, config: dict) -> bool:
    """
    Test forward pass for a specific model.
    
    Args:
        model_type: Type of model to test
        config: Configuration dictionary
        
    Returns:
        True if test passed, False otherwise
    """
    try:
        logger.info("Testing %s model forward pass...", model_type)
        
        # Update config for this model type
        test_config = config.copy()
        test_config['model']['type'] = model_type
        
        # Create model
        model = create_model_from_config(test_config)
        model.eval()
        
        # Create test input
        if model_type == 'unet':
            # U-Net expects fixed size input
            x = torch.randn(2, 3, 512, 512)
            
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (2, test_config['model']['unet']['n_classes'], 512, 512)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
        elif model_type == 'maskrcnn':
            # Mask R-CNN expects list of tensors
            images = [torch.randn(3, 800, 800), torch.randn(3, 600, 900)]
            
            with torch.no_grad():
                if hasattr(model, 'predict'):
                    predictions = model.predict(images)
                else:
                    predictions = model(images)
            
            assert isinstance(predictions, list), "Mask R-CNN should return list of predictions"
            assert len(predictions) == 2, f"Expected 2 predictions, got {len(predictions)}"
            
        elif model_type == 'yolov8':
            # YOLOv8 expects batch tensor
            x = torch.randn(2, 3, 640, 640)
            
            with torch.no_grad():
                output = model(x)
            
            # YOLOv8 output shape depends on implementation details
            assert output is not None, "YOLOv8 should return output"
        
        logger.info("✓ %s model forward pass successful", model_type)
        return True
        
    except Exception as e:
        logger.error("✗ %s model forward pass failed: %s", model_type, e)
        return False


def test_convenience_functions() -> bool:
    """Test convenience functions for creating models."""
    try:
        logger.info("Testing convenience functions...")
        
        # Test U-Net convenience function
        unet = create_unet_satellite_model(n_classes=3, use_attention=True)
        assert unet is not None
        
        # Test Mask R-CNN convenience function  
        maskrcnn = create_maskrcnn_satellite_model(num_classes=3)
        assert maskrcnn is not None
        
        # Test YOLOv8 convenience function
        yolov8 = create_yolov8_satellite_model(num_classes=3)
        assert yolov8 is not None
        
        logger.info("✓ Convenience functions work correctly")
        return True
        
    except Exception as e:
        logger.error("✗ Convenience functions failed: %s", e)
        return False


def test_model_factory() -> bool:
    """Test ModelFactory functionality."""
    try:
        logger.info("Testing ModelFactory...")
        
        # Test listing available models
        available_models = ModelFactory.list_available_models()
        expected_models = ['unet', 'maskrcnn', 'yolov8']
        
        for model_type in expected_models:
            assert model_type in available_models, f"Model {model_type} not available"
        
        logger.info("✓ ModelFactory works correctly")
        logger.info("  Available models: %s", available_models)
        return True
        
    except Exception as e:
        logger.error("✗ ModelFactory failed: %s", e)
        return False


def test_configuration_validation() -> bool:
    """Test configuration validation."""
    try:
        logger.info("Testing configuration validation...")
        
        # Test valid configuration
        valid_config = {
            'model': {
                'type': 'unet',
                'unet': {
                    'n_channels': 3,
                    'n_classes': 2
                }
            }
        }
        
        assert validate_model_config(valid_config)
        
        # Test invalid configuration (missing model section)
        try:
            invalid_config = {'data': {}}
            validate_model_config(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test invalid model type
        try:
            invalid_type_config = {
                'model': {
                    'type': 'invalid_model'
                }
            }
            validate_model_config(invalid_type_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        logger.info("✓ Configuration validation works correctly")
        return True
        
    except Exception as e:
        logger.error("✗ Configuration validation failed: %s", e)
        return False


def run_comprehensive_tests():
    """Run comprehensive tests for all CNN models."""
    logger.info("Starting comprehensive CNN model tests...")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return False
    
    # Test results
    results = {}
    
    # Available model types
    model_types = ['unet', 'maskrcnn', 'yolov8']
    
    # Test model factory
    results['model_factory'] = test_model_factory()
    
    # Test configuration validation
    results['config_validation'] = test_configuration_validation()
    
    # Test convenience functions
    results['convenience_functions'] = test_convenience_functions()
    
    # Test each model type
    for model_type in model_types:
        logger.info("\n" + "="*50)
        logger.info("Testing %s model", model_type.upper())
        logger.info("="*50)
        
        # Test model creation
        creation_result = test_model_creation(model_type, config)
        results[f'{model_type}_creation'] = creation_result
        
        # Test forward pass (only if creation succeeded)
        if creation_result:
            forward_result = test_model_forward_pass(model_type, config)
            results[f'{model_type}_forward'] = forward_result
        else:
            results[f'{model_type}_forward'] = False
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info("%-30s: %s", test_name, status)
        if result:
            passed += 1
    
    logger.info("-"*50)
    logger.info("Total: %d/%d tests passed (%.1f%%)", passed, total, (passed/total)*100)
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.warning("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    # Set device for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    
    # Run tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
