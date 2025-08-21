"""
Dataset loading utilities for satellite detection/segmentation.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset

class SatelliteDataset(Dataset):
    """PyTorch Dataset for satellite detection/segmentation."""
    
    def __init__(self, 
                 dataset_root: str,
                 split: str = 'train',
                 image_size: Optional[Tuple[int, int]] = None,
                 transforms=None):
        """
        Initialize satellite dataset.
        
        Args:
            dataset_root: Path to dataset root directory
            split: 'train' or 'test'
            image_size: Target image size (width, height) for resizing
            transforms: Optional transforms to apply
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        
        # Load annotations
        self.annotations = self._load_annotations()
        self.camera_params = self._load_camera_params()
        
    def _load_annotations(self) -> List[Dict]:
        """Load annotations for the specified split."""
        split_file = self.dataset_root / f'{self.split}.json'
        with open(split_file, 'r') as f:
            return json.load(f)
    
    def _load_camera_params(self) -> Dict:
        """Load camera intrinsic parameters."""
        camera_file = self.dataset_root / 'camera.json'
        with open(camera_file, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        annotation = self.annotations[idx]
        
        # Load image
        image_path = self.dataset_root / 'images' / annotation['filename']
        image = Image.open(image_path).convert('RGB')
        
        # Resize if specified
        original_size = image.size
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            scale_x = self.image_size[0] / original_size[0]
            scale_y = self.image_size[1] / original_size[1]
        else:
            scale_x = scale_y = 1.0
        
        # Process annotations
        bbox = np.array(annotation['bbox'], dtype=np.float32)
        keypoints = np.array(annotation['keypoints'], dtype=np.float32).reshape(-1, 3)
        pose_translation = np.array(annotation['r_xyz'], dtype=np.float32)
        pose_rotation = np.array(annotation['q_wxyz'], dtype=np.float32)
        
        # Scale bbox and keypoints if image was resized
        if self.image_size:
            bbox[0::2] *= scale_x  # x coordinates
            bbox[1::2] *= scale_y  # y coordinates
            keypoints[:, 0] *= scale_x  # x coordinates
            keypoints[:, 1] *= scale_y  # y coordinates
        
        # Convert to tensors
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'filename': annotation['filename'],
            'bbox': torch.from_numpy(bbox),
            'keypoints': torch.from_numpy(keypoints),
            'pose_translation': torch.from_numpy(pose_translation),
            'pose_rotation': torch.from_numpy(pose_rotation),
            'object_id': annotation['obj'],
            'image_size': original_size,
            'scale_factors': (scale_x, scale_y)
        }

def load_dataset_statistics(dataset_root: str) -> Dict[str, Any]:
    """Load comprehensive dataset statistics."""
    dataset_root = Path(dataset_root)
    
    # Load all data
    with open(dataset_root / 'gt.json', 'r') as f:
        gt_data = json.load(f)
    with open(dataset_root / 'train.json', 'r') as f:
        train_data = json.load(f)
    with open(dataset_root / 'test.json', 'r') as f:
        test_data = json.load(f)
    with open(dataset_root / 'camera.json', 'r') as f:
        camera_params = json.load(f)
    
    # Calculate statistics
    stats = {
        'total_images': len(gt_data),
        'train_size': len(train_data),
        'test_size': len(test_data),
        'image_resolution': (camera_params['Nu'], camera_params['Nv']),
        'camera_matrix': np.array(camera_params['cameraMatrix']),
        'focal_length': (camera_params['cameraMatrix'][0][0], camera_params['cameraMatrix'][1][1]),
        'principal_point': (camera_params['ccx'], camera_params['ccy']),
    }
    
    # Analyze annotations
    bbox_stats = analyze_bboxes(gt_data)
    keypoint_stats = analyze_keypoints(gt_data)
    pose_stats = analyze_poses(gt_data)
    
    stats.update({
        'bbox_stats': bbox_stats,
        'keypoint_stats': keypoint_stats,
        'pose_stats': pose_stats
    })
    
    return stats

def analyze_bboxes(annotations: List[Dict]) -> Dict[str, Any]:
    """Analyze bounding box statistics."""
    widths, heights, areas, aspect_ratios = [], [], [], []
    x_centers, y_centers = [], []
    
    for ann in annotations:
        bbox = ann['bbox']
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        aspect_ratios.append(w / h if h > 0 else 0)
        x_centers.append((x1 + x2) / 2)
        y_centers.append((y1 + y2) / 2)
    
    return {
        'width': {'mean': np.mean(widths), 'std': np.std(widths), 'min': min(widths), 'max': max(widths)},
        'height': {'mean': np.mean(heights), 'std': np.std(heights), 'min': min(heights), 'max': max(heights)},
        'area': {'mean': np.mean(areas), 'std': np.std(areas), 'min': min(areas), 'max': max(areas)},
        'aspect_ratio': {'mean': np.mean(aspect_ratios), 'std': np.std(aspect_ratios)},
        'centers': {'x': x_centers, 'y': y_centers}
    }

def analyze_keypoints(annotations: List[Dict]) -> Dict[str, Any]:
    """Analyze keypoint statistics."""
    visibility_stats = []
    keypoint_positions = [[] for _ in range(8)]
    
    for ann in annotations:
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        for i, (x, y, vis) in enumerate(keypoints):
            visibility_stats.append(vis)
            keypoint_positions[i].append((x, y, vis))
    
    visible_count = sum(1 for v in visibility_stats if v > 0.5)
    visibility_ratio = visible_count / len(visibility_stats)
    
    return {
        'total_keypoints': len(visibility_stats),
        'visible_count': visible_count,
        'visibility_ratio': visibility_ratio,
        'keypoint_positions': keypoint_positions
    }

def analyze_poses(annotations: List[Dict]) -> Dict[str, Any]:
    """Analyze pose statistics."""
    translations = []
    rotations = []
    distances = []
    
    for ann in annotations:
        translation = ann['r_xyz']
        rotation = ann['q_wxyz']
        
        translations.append(translation)
        rotations.append(rotation)
        distances.append(np.linalg.norm(translation))
    
    translations = np.array(translations)
    rotations = np.array(rotations)
    
    return {
        'translation': {
            'mean': np.mean(translations, axis=0),
            'std': np.std(translations, axis=0),
            'min': np.min(translations, axis=0),
            'max': np.max(translations, axis=0)
        },
        'distance': {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': min(distances),
            'max': max(distances)
        },
        'rotation_magnitude': {
            'mean': np.mean([np.linalg.norm(r) for r in rotations]),
            'std': np.std([np.linalg.norm(r) for r in rotations])
        }
    }

if __name__ == "__main__":
    # Test the dataset loader
    dataset_root = "/home/tanman/datasets/playground/NAPA2_Audacity_v2_training"
    
    # Load statistics
    stats = load_dataset_statistics(dataset_root)
    
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total images: {stats['total_images']:,}")
    print(f"Train/Test split: {stats['train_size']:,}/{stats['test_size']:,}")
    print(f"Image resolution: {stats['image_resolution'][0]}x{stats['image_resolution'][1]}")
    print(f"Focal length: {stats['focal_length'][0]:.1f}, {stats['focal_length'][1]:.1f}")
    print(f"Principal point: ({stats['principal_point'][0]:.1f}, {stats['principal_point'][1]:.1f})")
    
    print(f"\nBounding Box Statistics:")
    print(f"  Average size: {stats['bbox_stats']['width']['mean']:.1f} x {stats['bbox_stats']['height']['mean']:.1f}")
    print(f"  Size range: [{stats['bbox_stats']['width']['min']:.0f}-{stats['bbox_stats']['width']['max']:.0f}] x [{stats['bbox_stats']['height']['min']:.0f}-{stats['bbox_stats']['height']['max']:.0f}]")
    print(f"  Average area: {stats['bbox_stats']['area']['mean']:,.0f} pixels²")
    print(f"  Aspect ratio: {stats['bbox_stats']['aspect_ratio']['mean']:.2f} ± {stats['bbox_stats']['aspect_ratio']['std']:.2f}")
    
    print(f"\nKeypoint Statistics:")
    print(f"  Total keypoints: {stats['keypoint_stats']['total_keypoints']:,}")
    print(f"  Visible keypoints: {stats['keypoint_stats']['visibility_ratio']:.1%}")
    
    print(f"\nPose Statistics:")
    print(f"  Average distance: {stats['pose_stats']['distance']['mean']:.3f} ± {stats['pose_stats']['distance']['std']:.3f}")
    print(f"  Translation range: X[{stats['pose_stats']['translation']['min'][0]:.2f}, {stats['pose_stats']['translation']['max'][0]:.2f}], Y[{stats['pose_stats']['translation']['min'][1]:.2f}, {stats['pose_stats']['translation']['max'][1]:.2f}], Z[{stats['pose_stats']['translation']['min'][2]:.2f}, {stats['pose_stats']['translation']['max'][2]:.2f}]")
    
    # Test dataset loading
    print(f"\nTesting dataset loading...")
    train_dataset = SatelliteDataset(dataset_root, split='train', image_size=(640, 480))
    test_dataset = SatelliteDataset(dataset_root, split='test', image_size=(640, 480))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"\nSample data shapes:")
    print(f"  Image: {sample['image'].shape}")
    print(f"  Bbox: {sample['bbox'].shape}")
    print(f"  Keypoints: {sample['keypoints'].shape}")
    print(f"  Pose translation: {sample['pose_translation'].shape}")
    print(f"  Pose rotation: {sample['pose_rotation'].shape}")
    
    print("\nDataset loader test completed successfully! ✓")
