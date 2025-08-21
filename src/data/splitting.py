"""
Advanced data splitting strategies for satellite detection and segmentation.

This module implements stratified sampling based on pose diversity,
k-fold cross-validation, and other advanced splitting techniques.
"""

import json
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PoseBasedStratifiedSplitter:
    """
    Stratified data splitter based on satellite pose diversity.
    
    This class implements advanced splitting strategies that ensure balanced
    representation of different satellite poses across train/validation/test sets.
    """
    
    def __init__(self, n_pose_clusters: int = 10, random_state: int = 42):
        """
        Initialize the pose-based stratified splitter.
        
        Args:
            n_pose_clusters: Number of pose clusters for stratification
            random_state: Random seed for reproducibility
        """
        self.n_pose_clusters = n_pose_clusters
        self.random_state = random_state
        self.pose_clusters = None
        self.scaler = StandardScaler()
        
    def _extract_pose_features(self, data: List[Dict]) -> np.ndarray:
        """
        Extract pose features from dataset annotations.
        
        Args:
            data: List of dataset annotations
            
        Returns:
            Pose features as numpy array
        """
        features = []
        for item in data:
            # Extract rotation (r_xyz) and quaternion (q_wxyz) features
            r_xyz = item['r_xyz']
            q_wxyz = item['q_wxyz']
            
            # Combine rotation and quaternion into feature vector
            pose_vector = r_xyz + q_wxyz
            features.append(pose_vector)
            
        return np.array(features)
    
    def _cluster_poses(self, pose_features: np.ndarray) -> np.ndarray:
        """
        Cluster poses using K-means to create stratification groups.
        
        Args:
            pose_features: Pose feature vectors
            
        Returns:
            Cluster labels for each pose
        """
        # Normalize features
        pose_features_scaled = self.scaler.fit_transform(pose_features)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_pose_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(pose_features_scaled)
        
        self.pose_clusters = kmeans
        return cluster_labels
    
    def create_stratified_split(
        self, 
        data: List[Dict], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create stratified train/validation/test splits based on pose diversity.
        
        Args:
            data: Dataset annotations
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # Extract pose features and cluster
        pose_features = self._extract_pose_features(data)
        cluster_labels = self._cluster_poses(pose_features)
        
        logger.info("Created %d pose clusters for stratification", self.n_pose_clusters)
        
        # First split: separate test set
        indices = np.arange(len(data))
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=cluster_labels,
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        train_val_labels = cluster_labels[train_val_indices]
        val_size = val_ratio / (train_ratio + val_ratio)
        
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=self.random_state
        )
        
        # Create split datasets
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        logger.info("Created stratified splits: train=%d, val=%d, test=%d",
                   len(train_data), len(val_data), len(test_data))
        
        return train_data, val_data, test_data
    
    def get_cluster_distribution(self, data: List[Dict]) -> Dict[int, int]:
        """
        Get the distribution of samples across pose clusters.
        
        Args:
            data: Dataset annotations
            
        Returns:
            Dictionary mapping cluster ID to sample count
        """
        if self.pose_clusters is None:
            raise ValueError("Must fit clusters first using create_stratified_split")
        
        pose_features = self._extract_pose_features(data)
        pose_features_scaled = self.scaler.transform(pose_features)
        cluster_labels = self.pose_clusters.predict(pose_features_scaled)
        
        unique, counts = np.unique(cluster_labels, return_counts=True)
        return dict(zip(unique, counts))


class KFoldDataSplitter:
    """
    K-fold cross-validation splitter with pose-based stratification.
    """
    
    def __init__(self, n_folds: int = 5, n_pose_clusters: int = 10, random_state: int = 42):
        """
        Initialize K-fold splitter.
        
        Args:
            n_folds: Number of folds for cross-validation
            n_pose_clusters: Number of pose clusters for stratification
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.pose_splitter = PoseBasedStratifiedSplitter(n_pose_clusters, random_state)
        self.random_state = random_state
    
    def create_folds(self, data: List[Dict]) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Create K-fold splits with pose-based stratification.
        
        Args:
            data: Dataset annotations
            
        Returns:
            List of (train_data, val_data) tuples for each fold
        """
        # Extract pose features and cluster
        pose_features = self.pose_splitter._extract_pose_features(data)  # pylint: disable=protected-access
        cluster_labels = self.pose_splitter._cluster_poses(pose_features)  # pylint: disable=protected-access
        
        # Create stratified K-fold splitter
        skf = StratifiedKFold(
            n_splits=self.n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        fold_results = []
        indices = np.arange(len(data))
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(indices, cluster_labels)):
            train_data = [data[i] for i in train_indices]
            val_data = [data[i] for i in val_indices]
            
            logger.info("Fold %d: train=%d, val=%d", fold_idx + 1, len(train_data), len(val_data))
            fold_results.append((train_data, val_data))
        
        return fold_results


class DataSplitManager:
    """
    High-level manager for data splitting operations.
    """
    
    def __init__(self, dataset_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None):
        """
        Initialize data split manager.
        
        Args:
            dataset_path: Path to dataset directory
            config_path: Path to configuration file (default: configs/base_config.yaml)
        """
        self.dataset_path = Path(dataset_path)
        self.gt_data = None
        self.config = self._load_config(config_path)
        
    def load_ground_truth(self) -> List[Dict]:
        """
        Load ground truth annotations.
        
        Returns:
            List of ground truth annotations
        """
        gt_path = self.dataset_path / "gt.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        
        logger.info("Loaded %d ground truth annotations", len(self.gt_data))
        return self.gt_data
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Try to find config relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "base_config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning("Config file not found at %s, using defaults", config_path)
            return self._get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("Loaded configuration from %s", config_path)
        return config
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration values.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'cv_folds': 5,
                'n_pose_clusters': 10
            }
        }
    
    def create_enhanced_splits(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        n_pose_clusters: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Create enhanced stratified splits and save to files.
        
        Args:
            output_dir: Directory to save split files (default: dataset_path)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            n_pose_clusters: Number of pose clusters for stratification
            
        Returns:
            Dictionary with split data
        """
        if self.gt_data is None:
            self.load_ground_truth()
        
        # Use config values if parameters not provided
        data_config = self.config.get('data', {})
        if train_ratio is None:
            train_ratio = float(data_config.get('train_ratio', 0.7))
        if val_ratio is None:
            val_ratio = float(data_config.get('val_ratio', 0.15))
        if test_ratio is None:
            test_ratio = float(data_config.get('test_ratio', 0.15))
        if n_pose_clusters is None:
            n_pose_clusters = int(data_config.get('n_pose_clusters', 10))
        
        # Ensure data is loaded
        assert self.gt_data is not None, "Ground truth data must be loaded"
        
        if output_dir is None:
            output_dir = self.dataset_path
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create stratified splits
        splitter = PoseBasedStratifiedSplitter(
            n_pose_clusters=n_pose_clusters,
            random_state=42
        )
        
        train_data, val_data, test_data = splitter.create_stratified_split(
            self.gt_data, train_ratio, val_ratio, test_ratio
        )
        
        # Save splits to files
        split_results = {
            'train_stratified': train_data,
            'validation_stratified': val_data,
            'test_stratified': test_data
        }
        
        for split_name, split_data in split_results.items():
            output_path = output_dir / f"{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2)
            logger.info("Saved %s split to %s (%d samples)", split_name, output_path, len(split_data))
        
        # Log cluster distributions
        for split_name, split_data in split_results.items():
            distribution = splitter.get_cluster_distribution(split_data)
            logger.info("%s cluster distribution: %s", split_name, distribution)
        
        return split_results
    
    def create_kfold_splits(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        n_folds: Optional[int] = None,
        n_pose_clusters: Optional[int] = None
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Create K-fold cross-validation splits.
        
        Args:
            output_dir: Directory to save fold files (default: dataset_path/kfold)
            n_folds: Number of folds
            n_pose_clusters: Number of pose clusters for stratification
            
        Returns:
            List of (train_data, val_data) tuples for each fold
        """
        if self.gt_data is None:
            self.load_ground_truth()
        
        # Use config values if parameters not provided
        data_config = self.config.get('data', {})
        if n_folds is None:
            n_folds = int(data_config.get('cv_folds', 5))
        if n_pose_clusters is None:
            n_pose_clusters = int(data_config.get('n_pose_clusters', 10))
        
        if output_dir is None:
            output_dir = self.dataset_path / "kfold"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create K-fold splits
        kfold_splitter = KFoldDataSplitter(
            n_folds=n_folds,
            n_pose_clusters=n_pose_clusters,
            random_state=42
        )
        
        # Ensure gt_data is not None before passing
        assert self.gt_data is not None, "Ground truth data must be loaded"
        fold_results = kfold_splitter.create_folds(self.gt_data)
        
        # Save folds to files
        for fold_idx, (train_data, val_data) in enumerate(fold_results):
            fold_dir = output_dir / f"fold_{fold_idx + 1}"
            fold_dir.mkdir(exist_ok=True)
            
            train_path = fold_dir / "train.json"
            val_path = fold_dir / "validation.json"
            
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2)
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2)
            
            logger.info("Saved fold %d to %s", fold_idx + 1, fold_dir)
        
        return fold_results


def analyze_pose_diversity(data: List[Dict]) -> Dict[str, float]:
    """
    Analyze pose diversity in the dataset.
    
    Args:
        data: Dataset annotations
        
    Returns:
        Dictionary with diversity metrics
    """
    poses = []
    for item in data:
        r_xyz = item['r_xyz']
        q_wxyz = item['q_wxyz']
        poses.append(r_xyz + q_wxyz)
    
    poses = np.array(poses)
    
    # Calculate diversity metrics
    metrics = {
        'rotation_x_std': np.std(poses[:, 0]),
        'rotation_y_std': np.std(poses[:, 1]),
        'rotation_z_std': np.std(poses[:, 2]),
        'quaternion_w_std': np.std(poses[:, 3]),
        'quaternion_x_std': np.std(poses[:, 4]),
        'quaternion_y_std': np.std(poses[:, 5]),
        'quaternion_z_std': np.std(poses[:, 6]),
        'total_pose_variance': np.var(poses).sum()
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage with configuration
    logging.basicConfig(level=logging.INFO)
    
    dataset_path = "/home/tanman/datasets/playground/NAPA2_Audacity_v2_training"
    manager = DataSplitManager(dataset_path)
    
    print("Configuration-aware Data Splitting Demo")
    print("=" * 40)
    print(f"Train ratio: {manager.config['data']['train_ratio']}")
    print(f"Val ratio: {manager.config['data']['val_ratio']}")
    print(f"Test ratio: {manager.config['data']['test_ratio']}")
    print(f"CV folds: {manager.config['data']['cv_folds']}")
    print(f"Pose clusters: {manager.config['data']['n_pose_clusters']}")
    
    # Create enhanced stratified splits
    print("\nCreating enhanced stratified splits...")
    splits = manager.create_enhanced_splits()
    
    # Create K-fold splits  
    print("\nCreating K-fold cross-validation splits...")
    fold_results = manager.create_kfold_splits()
    
    # Analyze pose diversity
    print("\nAnalyzing pose diversity...")
    gt_data = manager.load_ground_truth()
    diversity = analyze_pose_diversity(gt_data)
    print("Pose diversity metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✅ Data splitting complete!")
    print(f"📁 Stratified splits: {len(splits)} sets created")
    print(f"📁 K-fold splits: {len(fold_results)} folds created")
    print(f"🎯 All splits use pose-based stratification for balanced representation")