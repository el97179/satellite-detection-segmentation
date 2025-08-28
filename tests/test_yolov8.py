"""
Test script for YOLOv8 model implementation.

Tests the YOLOv8 architecture for satellite detection including
forward pass validation, different configurations, and inference.
"""

import sys
import os
import torch
import torch.nn as nn
import yaml
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn.yolov8 import YOLOv8, create_yolov8_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_yolov8():
    """Test basic YOLOv8 functionality."""
    logger.info("Testing basic YOLOv8...")
    
    # Create YOLOv8 model
    model = YOLOv8(nc=2)  # 2 classes for satellite detection
    model.eval()
    
    # Test input
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        output = model(x)
    
    # Validate output exists
    assert output is not None, "YOLOv8 should return output"
    
    logger.info("✓ Basic YOLOv8 test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    logger.info(f"  Output type: {type(output)}")
    return True


def test_yolov8_different_sizes():
    """Test YOLOv8 with different input sizes."""
    logger.info("Testing YOLOv8 with different input sizes...")
    
    model = YOLOv8(nc=2)
    model.eval()
    
    input_sizes = [(640, 640), (416, 416), (832, 832)]
    
    for size in input_sizes:
        logger.info(f"Testing size: {size}")
        
        x = torch.randn(1, 3, size[0], size[1])
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None, f"Output should not be None for size {size}"
        logger.info(f"  ✓ Size {size} passed")
    
    return True


def test_yolov8_batch_processing():
    """Test YOLOv8 with batch inputs."""
    logger.info("Testing YOLOv8 batch processing...")
    
    model = YOLOv8(nc=2)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        x = torch.randn(batch_size, 3, 640, 640)
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None, f"Output should not be None for batch size {batch_size}"
        logger.info(f"  ✓ Batch size {batch_size} passed")
    
    return True


def test_yolov8_different_classes():
    """Test YOLOv8 with different number of classes."""
    logger.info("Testing YOLOv8 with different class numbers...")
    
    class_counts = [1, 2, 5, 10]
    
    for nc in class_counts:
        logger.info(f"Testing {nc} classes")
        
        model = YOLOv8(nc=nc)
        model.eval()
        
        x = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None, f"Output should not be None for {nc} classes"
        logger.info(f"  ✓ {nc} classes passed - Parameters: {model.get_num_parameters():,}")
    
    return True


def test_config_based_creation():
    """Test creating YOLOv8 from configuration file."""
    logger.info("Testing config-based YOLOv8 creation...")
    
    # Test configuration
    config = {
        'model': {
            'type': 'yolov8',
            'yolov8': {
                'num_classes': 2,
                'input_channels': 3,
                'architecture': None  # Use default
            }
        }
    }
    
    try:
        model = create_yolov8_model(config)
        model.eval()
        
        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(x)
        
        assert output is not None, "Output should not be None"
        
        logger.info("✓ Config-based YOLOv8 creation test passed")
        logger.info(f"  Parameters: {model.get_num_parameters():,}")
        return True
        
    except Exception as e:
        logger.error(f"Config-based creation failed: {e}")
        return False


def test_yolov8_training_mode():
    """Test YOLOv8 in training mode."""
    logger.info("Testing YOLOv8 training mode...")
    
    model = YOLOv8(nc=2)
    model.train()
    
    # Test input
    x = torch.randn(2, 3, 640, 640)
    
    # Forward pass in training mode
    output = model(x)
    
    # In training mode, output should be different format
    assert output is not None, "Training output should not be None"
    
    logger.info("✓ YOLOv8 training mode test passed")
    return True


def test_model_info():
    """Test YOLOv8 model information methods."""
    logger.info("Testing YOLOv8 model info...")
    
    model = YOLOv8(nc=2)
    
    # Test parameter counting
    param_count = model.get_num_parameters()
    assert param_count > 0, "Parameter count should be positive"
    
    # Test model info (if available)
    try:
        model.info(verbose=False)
        logger.info("✓ Model info method works")
    except Exception as e:
        logger.warning(f"Model info method failed: {e}")
    
    logger.info(f"✓ Model info test passed - Parameters: {param_count:,}")
    return True


def run_all_yolov8_tests():
    """Run all YOLOv8 tests."""
    logger.info("=" * 60)
    logger.info("RUNNING YOLOV8 COMPREHENSIVE TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Basic YOLOv8", test_basic_yolov8),
        ("Different Input Sizes", test_yolov8_different_sizes),
        ("Batch Processing", test_yolov8_batch_processing),
        ("Different Classes", test_yolov8_different_classes),
        ("Config-based Creation", test_config_based_creation),
        ("Training Mode", test_yolov8_training_mode),
        ("Model Info", test_model_info),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- {test_name} ---")
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("YOLOV8 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        logger.info(f"{test_name:<30}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All YOLOv8 tests passed!")
        return True
    else:
        logger.warning("⚠️ Some YOLOv8 tests failed")
        return False


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run tests
    success = run_all_yolov8_tests()
    
    if success:
        logger.info("\n✅ YOLOv8 implementation is ready for Issue #8!")
    else:
        logger.error("\n❌ YOLOv8 implementation needs fixes before PR")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)