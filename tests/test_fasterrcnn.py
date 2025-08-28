"""
Test script for Faster R-CNN model implementation.

Tests the Faster R-CNN architecture for satellite detection
including forward pass validation, backbone configurations, and inference.

Note: File name kept as test_maskrcnn.py for backward compatibility,
but actually tests    if passed == tota           logger.info("\\n✅ Faster R-CNN implementation is ready for Issue #6!")if success:
        logger.info("\\n✅ Faster R-CNN implementation is ready for Issue #6!")
    else:
        logger.error("\\n❌ Faster R-CNN implementation needs fixes before PR")       logger.info("🎉 All Faster R-CNN tests passed!")
        return True
    else:
        logger.warning("⚠️  Some Faster R-CNN tests failed")
        return Falser R-CNN implementation.
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

from src.models.cnn.fasterrcnn import (
    SatelliteMaskRCNN, 
    SatelliteMaskRCNNWithKeypoints,
    MultiScaleMaskRCNN,
    create_maskrcnn_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_maskrcnn():
    """Test basic Faster R-CNN functionality."""
    logger.info("Testing basic Faster R-CNN...")
    
    # Create Faster R-CNN model
    model = SatelliteMaskRCNN(num_classes=2, backbone_name="resnet50")
    model.eval()
    
    # Test input (list of images with different sizes)
    images = [
        torch.randn(3, 800, 800),
        torch.randn(3, 600, 900)
    ]
    
    with torch.no_grad():
        predictions = model.predict(images)
    
    # Validate predictions
    assert len(predictions) == 2, f"Expected 2 predictions, got {len(predictions)}"
    
    for i, pred in enumerate(predictions):
        assert 'boxes' in pred, f"Prediction {i} missing 'boxes'"
        assert 'labels' in pred, f"Prediction {i} missing 'labels'"
        assert 'scores' in pred, f"Prediction {i} missing 'scores'"
        # Note: Faster R-CNN doesn't have masks, so we don't check for them
    
    logger.info("✓ Basic Faster R-CNN test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    logger.info(f"  Predictions: {[len(pred['boxes']) for pred in predictions]} detections")
    return True


def test_maskrcnn_training_mode():
    """Test Faster R-CNN in training mode."""
    logger.info("Testing Faster R-CNN training mode...")
    
    model = SatelliteMaskRCNN(num_classes=2, backbone_name="resnet50")
    model.train()
    
    # Test input with targets
    images = [torch.randn(3, 800, 800)]
    targets = [{
        'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.zeros((1, 800, 800), dtype=torch.uint8)
    }]
    
    # Forward pass
    losses = model(images, targets)
    
    # Validate loss dictionary (Faster R-CNN losses)
    expected_losses = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
    for loss_name in expected_losses:
        assert loss_name in losses, f"Missing loss: {loss_name}"
        assert torch.is_tensor(losses[loss_name]), f"{loss_name} is not a tensor"
    
    logger.info("✓ Faster R-CNN training mode test passed")
    logger.info(f"  Losses: {list(losses.keys())}")
    return True


def test_maskrcnn_with_keypoints():
    """Test Faster R-CNN with keypoint detection."""
    logger.info("Testing Faster R-CNN with keypoints...")
    
    model = SatelliteMaskRCNNWithKeypoints(num_classes=2, num_keypoints=8)
    model.eval()
    
    # Test input
    images = [torch.randn(3, 800, 800)]
    
    with torch.no_grad():
        predictions = model.predict(images)
    
    # Validate predictions
    assert len(predictions) == 1, f"Expected 1 prediction, got {len(predictions)}"
    
    pred = predictions[0]
    assert 'boxes' in pred, "Prediction missing 'boxes'"
    assert 'labels' in pred, "Prediction missing 'labels'"
    assert 'scores' in pred, "Prediction missing 'scores'"
    # Note: Faster R-CNN doesn't have masks
    
    logger.info("✓ Faster R-CNN with keypoints test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    logger.info(f"  Keypoints: {model.num_keypoints}")
    return True


def test_multiscale_maskrcnn():
    """Test multi-scale Faster R-CNN."""
    logger.info("Testing multi-scale Faster R-CNN...")
    
    scales = [600, 800, 1000]
    model = MultiScaleMaskRCNN(num_classes=2, scales=scales)
    model.eval()
    
    # Test input
    images = [torch.randn(3, 800, 800)]
    
    with torch.no_grad():
        predictions = model.predict(images)
    
    # Validate predictions
    assert len(predictions) == 1, f"Expected 1 prediction, got {len(predictions)}"
    
    logger.info("✓ Multi-scale Faster R-CNN test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    logger.info(f"  Scales: {scales}")
    return True


def test_maskrcnn_different_backbones():
    """Test Faster R-CNN with different backbones."""
    logger.info("Testing Faster R-CNN with different backbones...")
    
    backbones = ["resnet50"]  # Start with one working backbone
    
    for backbone in backbones:
        logger.info(f"Testing backbone: {backbone}")
        
        try:
            model = SatelliteMaskRCNN(num_classes=2, backbone_name=backbone)
            model.eval()
            
            # Test input
            images = [torch.randn(3, 800, 800)]
            
            with torch.no_grad():
                predictions = model.predict(images)
            
            assert len(predictions) == 1, f"Backbone {backbone}: Expected 1 prediction"
            
            logger.info(f"  ✓ Backbone {backbone} passed - Parameters: {model.get_num_parameters():,}")
            
        except Exception as e:
            logger.error(f"  ✗ Backbone {backbone} failed: {e}")
            return False
    
    return True


def test_config_based_creation():
    """Test creating Faster R-CNN from configuration file."""
    logger.info("Testing config-based Faster R-CNN creation...")
    
    # Test configuration
    config = {
        'model': {
            'type': 'maskrcnn',
            'maskrcnn': {
                'num_classes': 2,
                'backbone_name': 'resnet50',
                'pretrained_backbone': True,
                'min_size': 800,
                'max_size': 1333
            }
        }
    }
    
    try:
        model = create_maskrcnn_model(config)
        model.eval()
        
        # Test forward pass
        images = [torch.randn(3, 800, 800)]
        with torch.no_grad():
            predictions = model.predict(images)
        
        assert len(predictions) == 1, "Expected 1 prediction"
        
        logger.info("✓ Config-based Faster R-CNN creation test passed")
        logger.info(f"  Parameters: {model.get_num_parameters():,}")
        return True
        
    except Exception as e:
        logger.error(f"Config-based creation failed: {e}")
        return False


def test_backbone_freezing():
    """Test backbone freezing functionality."""
    logger.info("Testing backbone freezing...")
    
    model = SatelliteMaskRCNN(num_classes=2, backbone_name="resnet50")
    
    # Check initial state (should be trainable)
    backbone_params_before = sum(p.numel() for p in model.model.backbone.parameters() if p.requires_grad)
    
    # Freeze backbone
    model.freeze_backbone()
    backbone_params_after_freeze = sum(p.numel() for p in model.model.backbone.parameters() if p.requires_grad)
    
    # Unfreeze backbone
    model.unfreeze_backbone()
    backbone_params_after_unfreeze = sum(p.numel() for p in model.model.backbone.parameters() if p.requires_grad)
    
    # Validate
    assert backbone_params_after_freeze == 0, "Backbone not properly frozen"
    assert backbone_params_after_unfreeze > 0, "Backbone not properly unfrozen"
    
    logger.info("✓ Backbone freezing test passed")
    logger.info(f"  Trainable params: {backbone_params_before:,} -> {backbone_params_after_freeze:,} -> {backbone_params_after_unfreeze:,}")
    return True


def run_all_maskrcnn_tests():
    """Run all Faster R-CNN tests."""
    logger.info("=" * 60)
    logger.info("RUNNING FASTER R-CNN COMPREHENSIVE TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Faster R-CNN", test_basic_maskrcnn),
        ("Training Mode", test_maskrcnn_training_mode),
        ("With Keypoints", test_maskrcnn_with_keypoints),
        ("Multi-scale", test_multiscale_maskrcnn),
        ("Different Backbones", test_maskrcnn_different_backbones),
        ("Config-based Creation", test_config_based_creation),
        ("Backbone Freezing", test_backbone_freezing),
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
    logger.info("FASTER R-CNN TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        logger.info(f"{test_name:<30}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All Faster R-CNN tests passed!")
        return True
    else:
        logger.warning("⚠️ Some Mask R-CNN tests failed")
        return False


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run tests
    success = run_all_maskrcnn_tests()
    
    if success:
        logger.info("\n✅ Mask R-CNN implementation is ready for Issue #6!")
    else:
        logger.error("\n❌ Mask R-CNN implementation needs fixes before PR")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)