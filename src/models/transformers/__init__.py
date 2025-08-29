"""
Transformer models for satellite detection and segmentation.

This module provides transformer-based architectures for satellite detection,
focusing on DETR (Detection Transformer) implementations.
"""

from .detr import (
    SatelliteDETR,
    SatelliteDETRWithKeypoints,
    MultiScaleDETR,
    create_detr_model
)

__all__ = [
    'SatelliteDETR',
    'SatelliteDETRWithKeypoints',
    'MultiScaleDETR',
    'create_detr_model'
]