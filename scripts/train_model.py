#!/usr/bin/env python
"""
End-to-end model training script.

Usage:
    python scripts/train_model.py --data-dir data/processed --output-dir artifacts/models
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train delay prediction model')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/models',
        help='Directory to save model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction for test set'
    )
    
    args = parser.parse_args()
    
    # Import here to allow running from repo root
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data.features import FeatureEngineer
    from src.models.train import ModelTrainer
    
    # 1. Load data
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # 2. Combine data
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file}")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(df)} rows")
    
    # 3. Engineer features
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    
    # Time-based split
    df = df.sort_values('feed_timestamp').reset_index(drop=True)
    split_idx = int(len(df) * (1 - args.test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create features
    train_df = engineer.create_features(train_df, reference_df=train_df)
    test_df = engineer.create_features(test_df, reference_df=train_df)
    
    feature_cols = engineer.get_feature_columns()
    
    # 4. Train model
    logger.info("Training model...")
    X_train = train_df[feature_cols].values
    y_train = train_df['delay_10plus'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['delay_10plus'].values
    
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, feature_names=feature_cols)
    
    # 5. Evaluate
    logger.info("Evaluating...")
    train_metrics = trainer.evaluate(X_train, y_train, 'train')
    test_metrics = trainer.evaluate(X_test, y_test, 'test')
    
    # 6. Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {'train': train_metrics, 'test': test_metrics}
    model_path = trainer.save(output_dir, metrics)
    
    # 7. Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"\nTest Metrics:")
    print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    print("\nTop 5 Important Features:")
    importance = trainer.get_feature_importance().head(5)
    for _, row in importance.iterrows():
        direction = "↑ increases" if row['coefficient'] > 0 else "↓ decreases"
        print(f"  {direction} {row['feature']}: {row['coefficient']:.4f}")


if __name__ == '__main__':
    main()
