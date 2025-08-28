"""
Test script for U-Net model implementation.

Tests the U-Net architecture for satellite segmentation including
forward pass validation, attention gates, and auxiliary outputs.
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

from src.models.cnn.unet import UNet, UNetWithAuxiliaryOutputs, create_unet_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_unet():
    """Test basic U-Net functionality."""
    logger.info("Testing basic U-Net...")
    
    # Create U-Net model
    model = UNet(n_channels=3, n_classes=2, depth=4)
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
    
    # Validate output shape
    expected_shape = (2, 2, 512, 512)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    logger.info("✓ Basic U-Net test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    return True


def test_unet_with_attention():
    """Test U-Net with attention gates."""
    logger.info("Testing U-Net with attention gates...")
    
    # Create U-Net with attention
    model = UNet(n_channels=3, n_classes=2, depth=4, use_attention=True)
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    # Validate output shape
    expected_shape = (2, 2, 256, 256)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    logger.info("✓ U-Net with attention test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    return True


def test_unet_auxiliary_outputs():
    """Test U-Net with auxiliary outputs."""
    logger.info("Testing U-Net with auxiliary outputs...")
    
    # Create U-Net with auxiliary outputs
    model = UNetWithAuxiliaryOutputs(n_channels=3, n_classes=2, depth=4)
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        main_output, aux_outputs = model(x)
    
    # Validate main output shape
    expected_main_shape = (2, 2, 512, 512)
    assert main_output.shape == expected_main_shape, f"Expected {expected_main_shape}, got {main_output.shape}"
    
    # Validate auxiliary outputs
    assert len(aux_outputs) == 3, f"Expected 3 auxiliary outputs, got {len(aux_outputs)}"
    for aux_out in aux_outputs:
        assert aux_out.shape == expected_main_shape, f"Auxiliary output shape mismatch: {aux_out.shape}"
    
    logger.info("✓ U-Net with auxiliary outputs test passed")
    logger.info(f"  Parameters: {model.get_num_parameters():,}")
    logger.info(f"  Auxiliary outputs: {len(aux_outputs)}")
    return True


def test_unet_different_configs():
    """Test U-Net with different configurations."""
    logger.info("Testing U-Net with different configurations...")
    
    configs = [
        {"n_channels": 3, "n_classes": 3, "depth": 3, "base_channels": 32},
        {"n_channels": 3, "n_classes": 5, "depth": 5, "base_channels": 64, "use_attention": True},
        {"n_channels": 1, "n_classes": 2, "depth": 4, "dropout_rate": 0.2},
    ]
    
    for i, config in enumerate(configs):
        logger.info(f"Testing configuration {i+1}: {config}")
        
        model = UNet(**config)
        model.eval()
        
        # Test with appropriate input channels
        x = torch.randn(1, config["n_channels"], 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (1, config["n_classes"], 256, 256)
        assert output.shape == expected_shape, f"Config {i+1}: Expected {expected_shape}, got {output.shape}"
        
        logger.info(f"  ✓ Configuration {i+1} passed - Parameters: {model.get_num_parameters():,}")
    
    return True


def test_config_based_creation():
    """Test creating U-Net from configuration file."""
    logger.info("Testing config-based U-Net creation...")
    
    # Test configuration
    config = {
        'model': {
            'type': 'unet',
            'unet': {
                'n_channels': 3,
                'n_classes': 2,
                'depth': 4,
                'use_attention': True,
                'dropout_rate': 0.1
            }
        }
    }
    
    try:
        model = create_unet_model(config)
        model.eval()
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (1, 2, 256, 256)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        logger.info("✓ Config-based U-Net creation test passed")
        logger.info(f"  Parameters: {model.get_num_parameters():,}")
        return True
        
    except Exception as e:
        logger.error(f"Config-based creation failed: {e}")
        return False


def test_gradient_flow():
    """Test gradient flow through U-Net."""
    logger.info("Testing gradient flow...")
    
    model = UNet(n_channels=3, n_classes=2, depth=4)
    model.train()
    
    # Create input and target
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    target = torch.randint(0, 2, (2, 256, 256)).long()
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are computed
    assert x.grad is not None, "Input gradients not computed"
    
    # Check model gradients
    has_gradients = any(param.grad is not None for param in model.parameters() if param.requires_grad)
    assert has_gradients, "Model gradients not computed"
    
    logger.info("✓ Gradient flow test passed")
    logger.info(f"  Loss: {loss.item():.4f}")
    return True


def run_all_unet_tests():
    """Run all U-Net tests."""
    logger.info("=" * 60)
    logger.info("RUNNING U-NET COMPREHENSIVE TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Basic U-Net", test_basic_unet),
        ("U-Net with Attention", test_unet_with_attention),
        ("U-Net with Auxiliary Outputs", test_unet_auxiliary_outputs),
        ("Different Configurations", test_unet_different_configs),
        ("Config-based Creation", test_config_based_creation),
        ("Gradient Flow", test_gradient_flow),
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
    logger.info("U-NET TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        logger.info(f"{test_name:<30}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All U-Net tests passed!")
        return True
    else:
        logger.warning("⚠️ Some U-Net tests failed")
        return False


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run tests
    success = run_all_unet_tests()
    
    if success:
        logger.info("\n✅ U-Net implementation is ready for Issue #4!")
    else:
        logger.error("\n❌ U-Net implementation needs fixes before PR")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)