"""
YOLOv8 implementation for satellite detection.

This module implements a simplified YOLOv8 architecture optimized for real-time satellite
detection with bounding box prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Auto padding calculation for 'same' padding."""
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True
    ):
        """
        Initialize convolution block.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            d: Dilation
            act: Add activation layer
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize bottleneck block."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck transformation."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """C2f block with 2 convolutions and n bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize C2f block."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C2f transformation."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize SPPF layer."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SPPF transformation."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class YOLOv8(nn.Module):
    """Simplified YOLOv8 model for satellite detection."""
    
    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        # Alias for tests that may pass ``nc`` (common YOLO arg name)
        nc: int = None,  # type: ignore[assignment]
    ):
        """
        Initialize YOLOv8 model for detection.
        
        Args:
            num_classes: Number of detection classes
            input_channels: Number of input channels
            width_mult: Width multiplier for channels
            depth_mult: Depth multiplier for layers
        """
        super().__init__()
        
        if nc is not None:
            num_classes = nc
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Base channel configuration (YOLOv8n)
        base_channels = [64, 128, 256, 512, 1024]
        channels = [max(round(c * width_mult), 1) for c in base_channels]
        
        # Backbone (simplified)
        self.backbone = nn.ModuleList([
            # Stem
            Conv(input_channels, channels[0], 6, 2, 2),  # P1/2
            Conv(channels[0], channels[1], 3, 2),         # P2/4
            C2f(channels[1], channels[1], max(round(3 * depth_mult), 1)),
            
            Conv(channels[1], channels[2], 3, 2),         # P3/8
            C2f(channels[2], channels[2], max(round(6 * depth_mult), 1)),
            
            Conv(channels[2], channels[3], 3, 2),         # P4/16
            C2f(channels[3], channels[3], max(round(6 * depth_mult), 1)),
            
            Conv(channels[3], channels[4], 3, 2),         # P5/32
            C2f(channels[4], channels[4], max(round(3 * depth_mult), 1)),
            SPPF(channels[4], channels[4], 5),
        ])
        
        # Neck (FPN + PAN)
        self.neck = nn.ModuleList([
            # Top-down pathway
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(channels[4] + channels[3], channels[3], max(round(3 * depth_mult), 1)),
            
            nn.Upsample(scale_factor=2, mode='nearest'), 
            C2f(channels[3] + channels[2], channels[2], max(round(3 * depth_mult), 1)),
            
            # Bottom-up pathway
            Conv(channels[2], channels[2], 3, 2),
            C2f(channels[2] + channels[3], channels[3], max(round(3 * depth_mult), 1)),
            
            Conv(channels[3], channels[3], 3, 2),
            C2f(channels[3] + channels[4], channels[4], max(round(3 * depth_mult), 1)),
        ])
        
        # Detection heads for 3 scales
        self.detection_heads = nn.ModuleList([
            DetectionHead(channels[2], num_classes),  # P3/8
            DetectionHead(channels[3], num_classes),  # P4/16  
            DetectionHead(channels[4], num_classes),  # P5/32
        ])
        
        self._initialize_weights()
        
        logger.info(f"YOLOv8 initialized: classes={num_classes}, parameters={self.get_num_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through YOLOv8.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of detection outputs at 3 scales
        """
        # Backbone forward pass
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:  # Save P3, P4, P5 features
                features.append(x)
        
        # Neck forward pass
        p5, p4, p3 = features[-1], features[-2], features[-3]
        
        # Top-down
        p5_up = self.neck[0](p5)  # Upsample P5
        p4_fused = torch.cat([p5_up, p4], dim=1)
        p4_out = self.neck[1](p4_fused)
        
        p4_up = self.neck[2](p4_out)  # Upsample P4
        p3_fused = torch.cat([p4_up, p3], dim=1)
        p3_out = self.neck[3](p3_fused)
        
        # Bottom-up
        p3_down = self.neck[4](p3_out)
        p4_fused2 = torch.cat([p3_down, p4_out], dim=1)
        p4_final = self.neck[5](p4_fused2)
        
        p4_down = self.neck[6](p4_final)
        p5_fused = torch.cat([p4_down, p5], dim=1)
        p5_final = self.neck[7](p5_fused)
        
        # Detection heads
        outputs = []
        detection_features = [p3_out, p4_final, p5_final]
        for i, head in enumerate(self.detection_heads):
            output = head(detection_features[i])
            outputs.append(output)
        
        return outputs
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DetectionHead(nn.Module):
    """Detection head for YOLOv8."""
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolution layers
        self.conv1 = Conv(in_channels, in_channels, 3)
        self.conv2 = Conv(in_channels, in_channels, 3)
        
        # Output layers: bbox (4) + objectness (1) + classes (num_classes)
        self.bbox_head = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
        self.cls_head = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through detection head."""
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Prediction outputs
        bbox_pred = self.bbox_head(x)  # (B, 4*A, H, W)
        obj_pred = self.obj_head(x)    # (B, A, H, W)  
        cls_pred = self.cls_head(x)    # (B, C*A, H, W)
        
        # Concatenate predictions: [bbox, obj, cls]
        B, _, H, W = bbox_pred.shape
        bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W)
        obj_pred = obj_pred.view(B, self.num_anchors, 1, H, W)
        cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W)
        
        output = torch.cat([bbox_pred, obj_pred, cls_pred], dim=2)  # (B, A, 5+C, H, W)
        return output


def create_yolov8_model(
    config: Optional[Dict] = None,
    num_classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    Create YOLOv8 model from configuration.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of detection classes
        **kwargs: Additional arguments
        
    Returns:
        Configured YOLOv8 model
    """
    if config is None:
        config = {}
    
    # Extract model parameters
    model_params = {
        'num_classes': config.get('num_classes', num_classes),
        'input_channels': config.get('input_channels', 3),
        'width_mult': config.get('width_mult', 1.0),
        'depth_mult': config.get('depth_mult', 1.0)
    }
    
    # Override with any additional kwargs
    model_params.update(kwargs)
    
    model = YOLOv8(**model_params)
    logger.info(f"Created YOLOv8 model with {model.get_num_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test YOLOv8 model
    model = YOLOv8(num_classes=1)  # 1 class: satellite
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    # Test with different input size
    x_small = torch.randn(1, 3, 416, 416)
    outputs_small = model(x_small)
    print(f"Small input shape: {x_small.shape}")
    for i, output in enumerate(outputs_small):
        print(f"Small output {i} shape: {output.shape}")