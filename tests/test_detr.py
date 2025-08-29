"""
Tests for DETR (Detection Transformer) model implementation.

This module contains comprehensive tests for the SatelliteDETR model,
including initialization, forward pass, training, and inference testing.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.models.transformers.detr import (
        SatelliteDETR, 
        SatelliteDETRWithKeypoints,
        MultiScaleDETR,
        create_detr_model
    )
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False


class TestSatelliteDETR:
    """Test cases for SatelliteDETR model."""
    
    @pytest.fixture
    def basic_detr_config(self):
        """Basic DETR configuration for testing."""
        return {
            'num_classes': 2,
            'num_queries': 50,
            'backbone_name': 'resnet50',
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'pretrained': False  # Avoid downloading pretrained weights in tests
        }
    
    @pytest.fixture
    def sample_images(self):
        """Sample input images for testing."""
        return [
            torch.randn(3, 640, 480),  # Different sizes to test flexibility
            torch.randn(3, 800, 600)
        ]
    
    @pytest.fixture
    def sample_targets(self):
        """Sample targets for training testing."""
        return [
            {
                'boxes': torch.tensor([[0.2, 0.2, 0.6, 0.6]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64)
            },
            {
                'boxes': torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]], dtype=torch.float32),
                'labels': torch.tensor([1, 1], dtype=torch.int64)
            }
        ]
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_detr_initialization(self, basic_detr_config):
        """Test DETR model initialization."""
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            # Mock the HuggingFace DETR model
            mock_model_instance = MagicMock()
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor.from_pretrained.return_value = MagicMock()
                
                model = SatelliteDETR(**basic_detr_config)
                
                assert model.num_classes == 2
                assert model.num_queries == 50
                assert model.backbone_name == 'resnet50'
                assert hasattr(model, 'config')
                assert hasattr(model, 'model')
                assert hasattr(model, 'image_processor')
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_detr_parameter_count(self, basic_detr_config):
        """Test parameter counting functionality."""
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            # Mock parameters method
            mock_model_instance.parameters.return_value = [
                torch.randn(100, 50, requires_grad=True),
                torch.randn(50, requires_grad=True)
            ]
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor.from_pretrained.return_value = MagicMock()
                
                model = SatelliteDETR(**basic_detr_config)
                
                # The actual parameter count depends on the mocked model
                param_count = model.get_num_parameters()
                assert isinstance(param_count, int)
                assert param_count >= 0
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_detr_forward_inference(self, basic_detr_config, sample_images):
        """Test DETR forward pass in inference mode."""
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            # Mock forward method to return expected output structure
            mock_output = MagicMock()
            mock_output.logits = torch.randn(2, 50, 2)  # [batch, queries, classes]
            mock_output.pred_boxes = torch.randn(2, 50, 4)  # [batch, queries, 4]
            mock_model_instance.return_value = mock_output
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor_instance = MagicMock()
                mock_processor_instance.return_value = {
                    'pixel_values': torch.randn(2, 3, 800, 800)
                }
                mock_processor.from_pretrained.return_value = mock_processor_instance
                
                model = SatelliteDETR(**basic_detr_config)
                model.eval()
                
                # Test inference
                with torch.no_grad():
                    predictions = model.predict(sample_images, threshold=0.1)
                
                assert isinstance(predictions, list)
                assert len(predictions) == 2  # One prediction per image
                
                for pred in predictions:
                    assert 'boxes' in pred
                    assert 'scores' in pred
                    assert 'labels' in pred
                    assert pred['boxes'].shape[1] == 4  # x_min, y_min, x_max, y_max
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_detr_forward_training(self, basic_detr_config, sample_images, sample_targets):
        """Test DETR forward pass in training mode."""
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            # Mock training forward to return loss dict
            mock_output = MagicMock()
            mock_output.loss = torch.tensor(1.5)
            mock_output.loss_dict = {
                'loss_ce': torch.tensor(0.5),
                'loss_bbox': torch.tensor(0.3),
                'loss_giou': torch.tensor(0.7)
            }
            mock_model_instance.return_value = mock_output
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor_instance = MagicMock()
                mock_processor_instance.return_value = {
                    'pixel_values': torch.randn(2, 3, 800, 800)
                }
                mock_processor.from_pretrained.return_value = mock_processor_instance
                
                model = SatelliteDETR(**basic_detr_config)
                model.train()
                
                # Test training forward
                outputs = model(sample_images, sample_targets)
                
                assert hasattr(outputs, 'loss') or isinstance(outputs, dict)
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_detr_freeze_unfreeze(self, basic_detr_config):
        """Test freezing and unfreezing backbone parameters."""
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            mock_backbone = MagicMock()
            mock_param = MagicMock()
            mock_param.requires_grad = True
            mock_backbone.parameters.return_value = [mock_param]
            mock_model_instance.model.backbone = mock_backbone
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor.from_pretrained.return_value = MagicMock()
                
                model = SatelliteDETR(**basic_detr_config)
                
                # Test freezing
                model.freeze_backbone()
                assert mock_param.requires_grad == False
                
                # Test unfreezing
                model.unfreeze_backbone()
                assert mock_param.requires_grad == True


class TestSatelliteDETRWithKeypoints:
    """Test cases for SatelliteDETR with keypoints."""
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_keypoint_detr_initialization(self):
        """Test DETR with keypoints initialization."""
        config = {
            'num_classes': 2,
            'num_keypoints': 8,
            'num_queries': 50,
            'pretrained': False
        }
        
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            mock_model_instance.config.d_model = 256
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor.from_pretrained.return_value = MagicMock()
                
                model = SatelliteDETRWithKeypoints(**config)
                
                assert model.num_keypoints == 8
                assert hasattr(model, 'keypoint_predictor')
                assert hasattr(model, 'visibility_predictor')
                assert model.config['num_keypoints'] == 8


class TestMultiScaleDETR:
    """Test cases for MultiScaleDETR."""
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_multiscale_detr_initialization(self):
        """Test multi-scale DETR initialization."""
        config = {
            'num_classes': 2,
            'scales': [480, 640, 800],
            'pretrained': False
        }
        
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor.from_pretrained.return_value = MagicMock()
                
                model = MultiScaleDETR(**config)
                
                assert model.scales == [480, 640, 800]


class TestDETRFactory:
    """Test cases for DETR model factory function."""
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_create_basic_detr(self):
        """Test creating basic DETR model from config."""
        config = {
            'model': {
                'detr': {
                    'num_classes': 3,
                    'num_queries': 75,
                    'backbone_name': 'resnet101',
                    'pretrained': False
                }
            }
        }
        
        with patch('src.models.transformers.detr.SatelliteDETR') as mock_detr:
            mock_instance = MagicMock()
            mock_instance.get_num_parameters.return_value = 41000000
            mock_detr.return_value = mock_instance
            
            model = create_detr_model(config)
            
            # Verify the model was created with correct parameters
            mock_detr.assert_called_once()
            call_args = mock_detr.call_args[1]  # Get keyword arguments
            assert call_args['num_classes'] == 3
            assert call_args['num_queries'] == 75
            assert call_args['backbone_name'] == 'resnet101'
            assert call_args['pretrained'] == False
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_create_keypoint_detr(self):
        """Test creating DETR with keypoints from config."""
        config = {
            'model': {
                'detr': {
                    'num_classes': 2,
                    'use_keypoints': True,
                    'num_keypoints': 8,
                    'pretrained': False
                }
            }
        }
        
        with patch('src.models.transformers.detr.SatelliteDETRWithKeypoints') as mock_detr:
            mock_instance = MagicMock()
            mock_instance.get_num_parameters.return_value = 45000000
            mock_detr.return_value = mock_instance
            
            model = create_detr_model(config)
            
            # Verify the keypoint model was created
            mock_detr.assert_called_once()
            call_args = mock_detr.call_args[1]
            assert call_args['num_keypoints'] == 8
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_create_multiscale_detr(self):
        """Test creating multi-scale DETR from config."""
        config = {
            'model': {
                'detr': {
                    'num_classes': 2,
                    'use_multiscale': True,
                    'scales': [400, 600, 800, 1000],
                    'pretrained': False
                }
            }
        }
        
        with patch('src.models.transformers.detr.MultiScaleDETR') as mock_detr:
            mock_instance = MagicMock()
            mock_instance.get_num_parameters.return_value = 42000000
            mock_detr.return_value = mock_instance
            
            model = create_detr_model(config)
            
            # Verify the multi-scale model was created
            mock_detr.assert_called_once()
            call_args = mock_detr.call_args[1]
            assert call_args['scales'] == [400, 600, 800, 1000]


class TestDETRTargetPreparation:
    """Test cases for DETR target preparation."""
    
    @pytest.mark.skipif(not DETR_AVAILABLE, reason="DETR dependencies not available")
    def test_prepare_targets_format(self):
        """Test target preparation converts to DETR format correctly."""
        config = {'num_classes': 2, 'pretrained': False}
        
        with patch('src.models.transformers.detr.DetrForObjectDetection') as mock_detr:
            mock_model_instance = MagicMock()
            mock_detr.return_value = mock_model_instance
            
            with patch('src.models.transformers.detr.DetrImageProcessor') as mock_processor:
                mock_processor.from_pretrained.return_value = MagicMock()
                
                model = SatelliteDETR(**config)
                
                # Test target preparation
                targets = [
                    {
                        'boxes': torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),  # [x_min, y_min, x_max, y_max]
                        'labels': torch.tensor([1], dtype=torch.int64)
                    }
                ]
                
                prepared = model._prepare_targets(targets)
                
                assert len(prepared) == 1
                assert 'boxes' in prepared[0]
                assert 'class_labels' in prepared[0]
                
                # Check box format conversion to center format
                boxes = prepared[0]['boxes']
                assert boxes.shape == (1, 4)  # [center_x, center_y, width, height]
                
                # Verify conversion: center_x = (0.1 + 0.3) / 2 = 0.2
                assert torch.allclose(boxes[0, 0], torch.tensor(0.2), atol=1e-6)
                # Verify conversion: center_y = (0.2 + 0.4) / 2 = 0.3
                assert torch.allclose(boxes[0, 1], torch.tensor(0.3), atol=1e-6)
                # Verify conversion: width = 0.3 - 0.1 = 0.2
                assert torch.allclose(boxes[0, 2], torch.tensor(0.2), atol=1e-6)
                # Verify conversion: height = 0.4 - 0.2 = 0.2
                assert torch.allclose(boxes[0, 3], torch.tensor(0.2), atol=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])