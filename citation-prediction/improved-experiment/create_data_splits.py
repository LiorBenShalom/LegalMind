#!/usr/bin/env python3
"""
Data Splitter for 5-Fold Cross-Validation
==========================================

Creates consistent train/val/test splits for 5-fold CV experiments.
Saves each fold as separate CSV files for reproducible experiments.

Usage:
    python create_data_splits.py
    
Output:
    data_splits/
    ├── original/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── fold_1/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── fold_2/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── ...
    └── fold_info.json
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, 
                 train_file: str,
                 val_file: str, 
                 test_file: str,
                 output_dir: str = "data_splits",
                 n_folds: int = 5,
                 random_seed: int = 42,
                 val_split_ratio: float = 0.2):
        """
        Initialize DataSplitter
        
        Args:
            train_file: Path to original training CSV
            val_file: Path to original validation CSV  
            test_file: Path to original test CSV
            output_dir: Directory to save split data
            n_folds: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
            val_split_ratio: Ratio for train/val split within each fold
        """
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.val_split_ratio = val_split_ratio
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seeds
        np.random.seed(random_seed)
        
    def load_data(self):
        """Load and combine all data for k-fold splitting"""
        logger.info("Loading original data files...")
        
        # Load datasets
        self.df_train = pd.read_csv(self.train_file)
        self.df_val = pd.read_csv(self.val_file)
        self.df_test = pd.read_csv(self.test_file)
        
        # Combine all data for k-fold CV
        self.df_full = pd.concat([self.df_train, self.df_val, self.df_test], ignore_index=True)
        
        logger.info(f"Loaded {len(self.df_train)} train + {len(self.df_val)} val + {len(self.df_test)} test = {len(self.df_full)} total samples")
        
        # Check label distribution
        label_dist = self.df_full['label'].value_counts().sort_index()
        logger.info(f"Label distribution: {dict(label_dist)}")
        logger.info(f"Class balance: {label_dist[1]/len(self.df_full)*100:.1f}% positive")
        
        return self.df_full
    
    def create_stratified_folds(self):
        """Create stratified k-fold splits"""
        logger.info(f"Creating {self.n_folds}-fold stratified splits...")
        
        # Create stratified k-fold splitter
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        
        # Get labels for stratification
        X = self.df_full.index.values
        y = self.df_full['label'].values
        
        self.fold_splits = []
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(kfold.split(X, y)):
            fold_info = {
                'fold': fold_idx + 1,
                'train_val_indices': train_val_indices.tolist(),
                'test_indices': test_indices.tolist(),
                'train_val_size': len(train_val_indices),
                'test_size': len(test_indices)
            }
            
            # Further split train_val into train and val
            train_val_data = self.df_full.iloc[train_val_indices]
            train_val_labels = train_val_data['label'].values
            
            # Stratified split for train/val
            train_indices_rel, val_indices_rel = train_test_split(
                range(len(train_val_indices)),
                test_size=self.val_split_ratio,
                stratify=train_val_labels,
                random_state=self.random_seed + fold_idx  # Different seed per fold
            )
            
            # Convert relative indices back to absolute indices
            train_indices_abs = train_val_indices[train_indices_rel]
            val_indices_abs = train_val_indices[val_indices_rel]
            
            fold_info.update({
                'train_indices': train_indices_abs.tolist(),
                'val_indices': val_indices_abs.tolist(),
                'train_size': len(train_indices_abs),
                'val_size': len(val_indices_abs)
            })
            
            # Check label distribution in each split
            train_labels = self.df_full.iloc[train_indices_abs]['label'].values
            val_labels = self.df_full.iloc[val_indices_abs]['label'].values
            test_labels = self.df_full.iloc[test_indices]['label'].values
            
            fold_info.update({
                'train_label_dist': np.bincount(train_labels).tolist(),
                'val_label_dist': np.bincount(val_labels).tolist(),
                'test_label_dist': np.bincount(test_labels).tolist(),
                'train_pos_ratio': np.mean(train_labels),
                'val_pos_ratio': np.mean(val_labels), 
                'test_pos_ratio': np.mean(test_labels)
            })
            
            self.fold_splits.append(fold_info)
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_indices_abs)}, Val={len(val_indices_abs)}, Test={len(test_indices)}")
            logger.info(f"  Positive ratios - Train: {np.mean(train_labels):.3f}, Val: {np.mean(val_labels):.3f}, Test: {np.mean(test_labels):.3f}")
        
        return self.fold_splits
    
    def save_fold_data(self):
        """Save each fold's data to separate CSV files"""
        logger.info("Saving fold data to CSV files...")
        
        for fold_info in self.fold_splits:
            fold_num = fold_info['fold']
            fold_dir = self.output_dir / f"fold_{fold_num}"
            fold_dir.mkdir(exist_ok=True)
            
            # Extract data for this fold
            train_data = self.df_full.iloc[fold_info['train_indices']].copy()
            val_data = self.df_full.iloc[fold_info['val_indices']].copy()
            test_data = self.df_full.iloc[fold_info['test_indices']].copy()
            
            # Add fold information
            train_data['fold'] = fold_num
            val_data['fold'] = fold_num  
            test_data['fold'] = fold_num
            
            train_data['split'] = 'train'
            val_data['split'] = 'val'
            test_data['split'] = 'test'
            
            # Save to CSV
            train_data.to_csv(fold_dir / "train.csv", index=False)
            val_data.to_csv(fold_dir / "val.csv", index=False)
            test_data.to_csv(fold_dir / "test.csv", index=False)
            
            logger.info(f"Saved fold {fold_num}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    def save_original_data(self):
        """Save original data splits for reference"""
        logger.info("Saving original data splits...")
        
        original_dir = self.output_dir / "original"
        original_dir.mkdir(exist_ok=True)
        
        # Copy original files
        self.df_train.to_csv(original_dir / "train.csv", index=False)
        self.df_val.to_csv(original_dir / "val.csv", index=False)
        self.df_test.to_csv(original_dir / "test.csv", index=False)
        
        # Also save combined data
        self.df_full.to_csv(original_dir / "combined.csv", index=False)
        
        logger.info(f"Saved original splits to {original_dir}")
    
    def save_fold_info(self):
        """Save fold information and metadata"""
        fold_info_file = self.output_dir / "fold_info.json"
        
        metadata = {
            'n_folds': self.n_folds,
            'random_seed': self.random_seed,
            'val_split_ratio': self.val_split_ratio,
            'total_samples': len(self.df_full),
            'original_files': {
                'train': self.train_file,
                'val': self.val_file,
                'test': self.test_file
            },
            'fold_splits': self.fold_splits
        }
        
        with open(fold_info_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved fold information to {fold_info_file}")
    
    def create_summary_stats(self):
        """Create summary statistics across all folds"""
        logger.info("Creating summary statistics...")
        
        # Calculate statistics across folds
        stats = {
            'avg_train_size': np.mean([fold['train_size'] for fold in self.fold_splits]),
            'avg_val_size': np.mean([fold['val_size'] for fold in self.fold_splits]),
            'avg_test_size': np.mean([fold['test_size'] for fold in self.fold_splits]),
            'avg_train_pos_ratio': np.mean([fold['train_pos_ratio'] for fold in self.fold_splits]),
            'avg_val_pos_ratio': np.mean([fold['val_pos_ratio'] for fold in self.fold_splits]),
            'avg_test_pos_ratio': np.mean([fold['test_pos_ratio'] for fold in self.fold_splits]),
            'std_train_pos_ratio': np.std([fold['train_pos_ratio'] for fold in self.fold_splits]),
            'std_val_pos_ratio': np.std([fold['val_pos_ratio'] for fold in self.fold_splits]),
            'std_test_pos_ratio': np.std([fold['test_pos_ratio'] for fold in self.fold_splits])
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 FOLD SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Average sizes: Train={stats['avg_train_size']:.0f}, Val={stats['avg_val_size']:.0f}, Test={stats['avg_test_size']:.0f}")
        print(f"Positive class ratios:")
        print(f"  Train: {stats['avg_train_pos_ratio']:.3f} ± {stats['std_train_pos_ratio']:.3f}")
        print(f"  Val:   {stats['avg_val_pos_ratio']:.3f} ± {stats['std_val_pos_ratio']:.3f}")
        print(f"  Test:  {stats['avg_test_pos_ratio']:.3f} ± {stats['std_test_pos_ratio']:.3f}")
        
        # Save stats
        stats_file = self.output_dir / "summary_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def run(self):
        """Execute the complete data splitting pipeline"""
        logger.info("🚀 Starting data splitting pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create stratified folds
        self.create_stratified_folds()
        
        # Step 3: Save fold data
        self.save_fold_data()
        
        # Step 4: Save original data
        self.save_original_data()
        
        # Step 5: Save fold information
        self.save_fold_info()
        
        # Step 6: Create summary statistics
        self.create_summary_stats()
        
        logger.info("✅ Data splitting pipeline completed successfully!")
        logger.info(f"📁 Data splits saved to: {self.output_dir.absolute()}")
        
        return self.fold_splits

def main():
    """Main execution function"""
    # File paths - UPDATE THESE TO YOUR PATHS
    train_file = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_train.csv"
    val_file = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_val.csv"
    test_file = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_test.csv"
    
    # Create data splitter
    splitter = DataSplitter(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        output_dir="data_splits_5fold",
        n_folds=5,
        random_seed=42,
        val_split_ratio=0.2
    )
    
    # Run the pipeline
    fold_splits = splitter.run()
    
    # Print final summary
    print(f"\n🎯 PIPELINE COMPLETE!")
    print(f"Created {len(fold_splits)} folds with consistent stratified splits")
    print(f"Each fold maintains similar class distributions")
    print(f"Ready for reproducible cross-validation experiments!")

if __name__ == "__main__":
    main()