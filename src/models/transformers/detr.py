"""
DETR (Detection Transformer) implementation for satellite detection.

This module implements a DETR architecture optimized for satellite detection,
leveraging HuggingFace transformers for end-to-end object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    from transformers import DetrForObjectDetection, DetrConfig, DetrImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DetrForObjectDetection = None
    DetrConfig = None
    DetrImageProcessor = None

logger = logging.getLogger(__name__)


class SatelliteDETR(nn.Module):
    """
    DETR (Detection Transformer) implementation for satellite detection.
    
    Features:
    - Transformer-based end-to-end object detection
    - No post-processing (NMS) required
    - Bipartite matching for training
    - Configurable number of object queries
    - Support for different backbone architectures
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # Background + satellite
        num_queries: int = 100,  # Number of object queries
        backbone_name: str = "resnet50",
        d_model: int = 256,  # Transformer hidden dimension
        nhead: int = 8,  # Number of attention heads
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1,  # Loss coefficient for empty object class
        pretrained: bool = True,
        **kwargs
    ):
        """
        Initialize DETR model.
        
        Args:
            num_classes: Number of classes (including background)
            num_queries: Number of object queries
            backbone_name: Backbone architecture name
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            activation: Activation function
            bbox_loss_coef: Bounding box loss coefficient
            giou_loss_coef: GIoU loss coefficient
            eos_coef: End-of-sequence loss coefficient
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers is required for DETR. "
                "Install with: pip install transformers"
            )
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone_name = backbone_name
        
        # Create DETR configuration
        config = DetrConfig(
            num_labels=num_classes,
            num_queries=num_queries,
            d_model=d_model,
            encoder_attention_heads=nhead,
            decoder_attention_heads=nhead,
            encoder_layers=num_encoder_layers,
            decoder_layers=num_decoder_layers,
            encoder_ffn_dim=dim_feedforward,
            decoder_ffn_dim=dim_feedforward,
            dropout=dropout,
            activation_function=activation,
            bbox_loss_coef=bbox_loss_coef,
            giou_loss_coef=giou_loss_coef,
            eos_coef=eos_coef,
            backbone=backbone_name,
            use_pretrained_backbone=pretrained,
            **kwargs
        )
        
        # Initialize DETR model
        if pretrained:
            try:
                # Try to load pretrained DETR from HuggingFace
                self.model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50",
                    config=config,
                    ignore_mismatched_sizes=True
                )
                # Replace classification head for our number of classes
                self.model.class_labels_classifier = nn.Linear(
                    config.d_model, num_classes
                )
                logger.info("Loaded pretrained DETR from HuggingFace")
            except Exception as e:
                logger.warning(f"Failed to load pretrained DETR: {e}")
                self.model = DetrForObjectDetection(config)
        else:
            self.model = DetrForObjectDetection(config)
        
        # Image processor for preprocessing
        self.image_processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        
        # Store configuration
        self.config = {
            'num_classes': num_classes,
            'num_queries': num_queries,
            'backbone_name': backbone_name,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
        }
        
        logger.info(
            "Initialized DETR: backbone=%s, classes=%d, queries=%d, parameters=%d",
            backbone_name, num_classes, num_queries, self.get_num_parameters()
        )
    
    def forward(
        self, 
        images: Union[List[Tensor], Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Forward pass through DETR.
        
        Args:
            images: Input images (batch of tensors or list of tensors)
            targets: Target dictionaries (for training)
            
        Returns:
            During training: Dictionary of losses
            During inference: Model outputs with logits and boxes
        """
        # Convert images to proper format if needed
        if isinstance(images, list):
            # Process list of images
            batch_size = len(images)
            processed = self.image_processor(
                images, 
                return_tensors="pt",
                do_resize=True,
                size={"height": 800, "width": 800}
            )
            pixel_values = processed["pixel_values"]
        else:
            # Single batch tensor
            pixel_values = images
            batch_size = images.shape[0]
        
        # Prepare labels for training
        if targets is not None:
            labels = self._prepare_targets(targets)
            outputs = self.model(pixel_values=pixel_values, labels=labels)
        else:
            outputs = self.model(pixel_values=pixel_values)
        
        return outputs
    
    def _prepare_targets(self, targets: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        """
        Prepare targets for DETR format.
        
        Args:
            targets: List of target dictionaries
            
        Returns:
            Prepared targets in DETR format
        """
        prepared_targets = []
        
        for target in targets:
            prepared_target = {}
            
            # Convert boxes to DETR format (center_x, center_y, width, height) normalized
            boxes = target['boxes']  # Expected format: [x_min, y_min, x_max, y_max]
            
            # Convert to center format and normalize
            # Note: This assumes boxes are already normalized to [0, 1]
            center_x = (boxes[:, 0] + boxes[:, 2]) / 2
            center_y = (boxes[:, 1] + boxes[:, 3]) / 2
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            
            prepared_target['boxes'] = torch.stack([center_x, center_y, width, height], dim=1)
            
            # Labels (convert to 0-indexed, subtract 1 if needed)
            if 'labels' in target:
                labels = target['labels']
                # Ensure labels are 0-indexed (background=0, satellite=1)
                prepared_target['class_labels'] = labels - 1 if labels.min() > 0 else labels
            
            prepared_targets.append(prepared_target)
        
        return prepared_targets
    
    def predict(
        self, 
        images: Union[List[Tensor], Tensor],
        threshold: float = 0.5
    ) -> List[Dict[str, Tensor]]:
        """
        Make predictions on images.
        
        Args:
            images: Input images
            threshold: Confidence threshold for predictions
            
        Returns:
            List of predictions per image
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.forward(images)
        
        # Convert outputs to standard format
        predictions = self._postprocess_predictions(outputs, threshold)
        
        return predictions
    
    def _postprocess_predictions(
        self, 
        outputs, 
        threshold: float = 0.5
    ) -> List[Dict[str, Tensor]]:
        """
        Post-process DETR outputs to standard format.
        
        Args:
            outputs: DETR model outputs
            threshold: Confidence threshold
            
        Returns:
            List of predictions in standard format
        """
        predictions = []
        
        # Get logits and boxes
        logits = outputs.logits  # [batch_size, num_queries, num_classes]
        boxes = outputs.pred_boxes  # [batch_size, num_queries, 4]
        
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            # Get probabilities for this image
            probs = F.softmax(logits[i], dim=-1)  # [num_queries, num_classes]
            
            # Get maximum probability and predicted class for each query
            max_probs, pred_labels = probs.max(dim=-1)
            
            # Filter out background class (assuming class 0 is background)
            # and apply confidence threshold
            keep = (pred_labels > 0) & (max_probs > threshold)
            
            if keep.sum() > 0:
                pred_boxes = boxes[i][keep]  # [num_detections, 4]
                pred_scores = max_probs[keep]  # [num_detections]
                pred_classes = pred_labels[keep]  # [num_detections]
                
                # Convert boxes from center format to corner format
                # DETR outputs: [center_x, center_y, width, height]
                # Standard format: [x_min, y_min, x_max, y_max]
                center_x, center_y, width, height = pred_boxes.unbind(1)
                x_min = center_x - width / 2
                y_min = center_y - height / 2
                x_max = center_x + width / 2
                y_max = center_y + height / 2
                
                pred_boxes_corner = torch.stack([x_min, y_min, x_max, y_max], dim=1)
                
                predictions.append({
                    'boxes': pred_boxes_corner,
                    'scores': pred_scores,
                    'labels': pred_classes
                })
            else:
                # No detections
                predictions.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty((0,)),
                    'labels': torch.empty((0,), dtype=torch.long)
                })
        
        return predictions
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.model.model.backbone.parameters():
            param.requires_grad = False
        logger.info("Frozen backbone parameters")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.model.model.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfrozen backbone parameters")


class SatelliteDETRWithKeypoints(SatelliteDETR):
    """
    Extended DETR with keypoint detection for satellite pose estimation.
    
    This extends the base DETR to also predict keypoints for 6DoF pose estimation.
    """
    
    def __init__(
        self, 
        num_keypoints: int = 8,
        keypoint_loss_coef: float = 1.0,
        **kwargs
    ):
        """
        Initialize DETR with keypoint detection.
        
        Args:
            num_keypoints: Number of keypoints to detect
            keypoint_loss_coef: Keypoint loss coefficient
            **kwargs: Arguments passed to base class
        """
        super().__init__(**kwargs)
        
        self.num_keypoints = num_keypoints
        
        # Add keypoint prediction head
        d_model = self.model.config.d_model
        self.keypoint_predictor = nn.Linear(d_model, num_keypoints * 2)  # x, y coordinates
        
        # Add visibility predictor
        self.visibility_predictor = nn.Linear(d_model, num_keypoints)  # visibility flags
        
        self.config.update({
            'num_keypoints': num_keypoints,
            'keypoint_loss_coef': keypoint_loss_coef
        })
        
        logger.info("Added keypoint detection: %d keypoints", num_keypoints)
    
    def forward(
        self, 
        images: Union[List[Tensor], Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Forward pass with keypoint prediction.
        
        Args:
            images: Input images
            targets: Target dictionaries (including keypoints)
            
        Returns:
            Outputs including keypoint predictions
        """
        # Get base DETR outputs
        outputs = super().forward(images, targets)
        
        # Get decoder hidden states for keypoint prediction
        decoder_hidden_states = self.model.model.decoder.last_hidden_state
        
        # Predict keypoints and visibility
        keypoint_coords = self.keypoint_predictor(decoder_hidden_states)  # [batch, queries, num_kpts*2]
        keypoint_visibility = self.visibility_predictor(decoder_hidden_states)  # [batch, queries, num_kpts]
        
        # Reshape keypoint coordinates
        batch_size, num_queries = keypoint_coords.shape[:2]
        keypoint_coords = keypoint_coords.view(batch_size, num_queries, self.num_keypoints, 2)
        
        # Add to outputs
        if hasattr(outputs, 'pred_keypoints'):
            outputs.pred_keypoints = keypoint_coords
            outputs.pred_keypoint_visibility = torch.sigmoid(keypoint_visibility)
        else:
            # For custom outputs dict
            if isinstance(outputs, dict):
                outputs['pred_keypoints'] = keypoint_coords
                outputs['pred_keypoint_visibility'] = torch.sigmoid(keypoint_visibility)
        
        return outputs


class MultiScaleDETR(SatelliteDETR):
    """
    Multi-scale DETR for handling satellites at different scales.
    
    Uses multiple image scales during training and inference
    to improve detection of satellites at various sizes.
    """
    
    def __init__(
        self,
        scales: List[int] = [480, 640, 800, 960],
        **kwargs
    ):
        """
        Initialize multi-scale DETR.
        
        Args:
            scales: List of image scales to use
            **kwargs: Arguments passed to base class
        """
        super().__init__(**kwargs)
        self.scales = scales
        
        logger.info("Initialized multi-scale DETR with scales: %s", scales)
    
    def forward(
        self, 
        images: Union[List[Tensor], Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Multi-scale forward pass.
        
        During training, randomly selects a scale.
        During inference, can optionally use multiple scales.
        """
        if self.training and len(self.scales) > 1:
            # Randomly select a scale during training
            import random
            selected_scale = random.choice(self.scales)
            
            # Resize images to selected scale
            if isinstance(images, list):
                processed = self.image_processor(
                    images, 
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": selected_scale, "width": selected_scale}
                )
                pixel_values = processed["pixel_values"]
            else:
                # Resize tensor images
                pixel_values = F.interpolate(
                    images, 
                    size=(selected_scale, selected_scale), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Forward with resized images
            if targets is not None:
                labels = self._prepare_targets(targets)
                outputs = self.model(pixel_values=pixel_values, labels=labels)
            else:
                outputs = self.model(pixel_values=pixel_values)
            
            return outputs
        else:
            return super().forward(images, targets)


def create_detr_model(config: dict) -> nn.Module:
    """
    Create DETR model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured DETR model
    """
    model_config = config.get('model', {})
    detr_config = model_config.get('detr', {})
    
    # Default parameters
    params = {
        'num_classes': detr_config.get('num_classes', 2),
        'num_queries': detr_config.get('num_queries', 100),
        'backbone_name': detr_config.get('backbone_name', 'resnet50'),
        'd_model': detr_config.get('d_model', 256),
        'nhead': detr_config.get('nhead', 8),
        'num_encoder_layers': detr_config.get('num_encoder_layers', 6),
        'num_decoder_layers': detr_config.get('num_decoder_layers', 6),
        'dim_feedforward': detr_config.get('dim_feedforward', 2048),
        'dropout': detr_config.get('dropout', 0.1),
        'bbox_loss_coef': detr_config.get('bbox_loss_coef', 5.0),
        'giou_loss_coef': detr_config.get('giou_loss_coef', 2.0),
        'eos_coef': detr_config.get('eos_coef', 0.1),
        'pretrained': detr_config.get('pretrained', True)
    }
    
    # Check for special variants
    use_keypoints = detr_config.get('use_keypoints', False)
    use_multiscale = detr_config.get('use_multiscale', False)
    
    if use_keypoints:
        num_keypoints = detr_config.get('num_keypoints', 8)
        keypoint_loss_coef = detr_config.get('keypoint_loss_coef', 1.0)
        model = SatelliteDETRWithKeypoints(
            num_keypoints=num_keypoints,
            keypoint_loss_coef=keypoint_loss_coef,
            **params
        )
    elif use_multiscale:
        scales = detr_config.get('scales', [480, 640, 800, 960])
        model = MultiScaleDETR(scales=scales, **params)
    else:
        model = SatelliteDETR(**params)
    
    logger.info("Created DETR model with %d parameters", model.get_num_parameters())
    return model


if __name__ == "__main__":
    # Example usage and testing
    import os
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Reduce transformers logging
    
    logging.basicConfig(level=logging.INFO)
    
    if not TRANSFORMERS_AVAILABLE:
        print("HuggingFace transformers not available. Skipping tests.")
        exit(1)
    
    # Test basic DETR
    print("Testing basic DETR...")
    detection_model = SatelliteDETR(num_classes=2, num_queries=50)
    
    print(f"Model parameters: {detection_model.get_num_parameters():,}")
    
    # Test input (list of images)
    images = [torch.randn(3, 800, 800), torch.randn(3, 600, 900)]
    
    # Test inference mode
    print("Testing inference...")
    detection_model.eval()
    predictions = detection_model.predict(images, threshold=0.1)
    print(f"Number of predictions: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"Image {i}: {len(pred['boxes'])} detections")
    
    # Test training mode with targets
    print("Testing training mode...")
    targets = [
        {
            'boxes': torch.tensor([[0.2, 0.2, 0.4, 0.4]], dtype=torch.float32),  # Normalized
            'labels': torch.tensor([1], dtype=torch.int64),
        },
        {
            'boxes': torch.tensor([[0.3, 0.3, 0.5, 0.5]], dtype=torch.float32),  # Normalized
            'labels': torch.tensor([1], dtype=torch.int64),
        }
    ]
    
    detection_model.train()
    try:
        losses = detection_model(images, targets)
        print(f"Training completed. Loss keys: {list(losses.keys()) if hasattr(losses, 'keys') else 'N/A'}")
    except Exception as e:
        print(f"Training test failed: {e}")
    
    print("DETR model test completed successfully!")