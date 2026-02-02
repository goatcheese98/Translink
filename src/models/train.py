"""
Model Training Pipeline

Trains and evaluates delay prediction models.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate delay prediction models."""
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: 'logistic_regression' or 'xgboost' (future).
            class_weight: How to handle class imbalance.
            random_state: Random seed.
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = None
    
    def create_model(self) -> LogisticRegression:
        """Create the model instance."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                class_weight=self.class_weight,
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'ModelTrainer':
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            feature_names: Names of features (for interpretability).
            
        Returns:
            Self for method chaining.
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        
        logger.info(f"Training {self.model_type} on {len(X_train)} samples")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Class distribution: {np.bincount(y_train.astype(int))}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        self.model = self.create_model()
        self.model.fit(X_train_scaled, y_train)
        
        logger.info("Training complete")
        return self
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = 'test'
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            X: Features.
            y: True labels.
            dataset_name: Name for logging.
            
        Returns:
            Dictionary of metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int((~y.astype(bool)).sum())
        }
        
        logger.info(f"{dataset_name} metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients for logistic regression).
        
        Returns:
            DataFrame with feature names and coefficients.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        coeffs = self.model.coef_[0]
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coeffs,
            'abs_coefficient': np.abs(coeffs)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def save(
        self,
        output_dir: Union[str, Path],
        metrics: Optional[Dict] = None
    ) -> Path:
        """
        Save model artifacts.
        
        Args:
            output_dir: Directory to save to.
            metrics: Optional metrics to save alongside.
            
        Returns:
            Path to saved model.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model bundle
        model_bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        model_path = output_dir / f'model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        # Save metrics separately
        if metrics:
            metrics_path = output_dir / f'metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save feature importance
        importance = self.get_feature_importance()
        importance_path = output_dir / f'feature_importance_{timestamp}.csv'
        importance.to_csv(importance_path, index=False)
        
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> 'ModelTrainer':
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model.
            
        Returns:
            Loaded ModelTrainer instance.
        """
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        trainer = cls(
            model_type=bundle.get('model_type', 'logistic_regression')
        )
        trainer.model = bundle['model']
        trainer.scaler = bundle['scaler']
        trainer.feature_names = bundle['feature_names']
        
        logger.info(f"Loaded model from {model_path}")
        return trainer
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Features.
            
        Returns:
            Tuple of (predictions, probabilities).
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        return y_pred, y_proba


def train_from_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'delay_10plus',
    test_size: float = 0.2,
    output_dir: Optional[str] = None
) -> Tuple[ModelTrainer, Dict]:
    """
    Train model from DataFrame.
    
    Args:
        df: DataFrame with features and target.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        test_size: Fraction for test set.
        output_dir: Where to save model (optional).
        
    Returns:
        Tuple of (trained trainer, metrics dict).
    """
    from sklearn.model_selection import train_test_split
    
    # Sort by time if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Split
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, feature_names=feature_cols)
    
    # Evaluate
    train_metrics = trainer.evaluate(X_train, y_train, 'train')
    test_metrics = trainer.evaluate(X_test, y_test, 'test')
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'feature_importance': trainer.get_feature_importance().to_dict()
    }
    
    # Save if requested
    if output_dir:
        trainer.save(output_dir, metrics)
    
    return trainer, metrics


def main():
    """Run training locally."""
    import argparse
    from src.data.features import FeatureEngineer
    
    parser = argparse.ArgumentParser(description='Train delay prediction model')
    parser.add_argument(
        'input',
        type=str,
        help='Input CSV file with features'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default='artifacts/models',
        help='Output directory for model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction for test set'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Engineer features if not already done
    if 'hour' not in df.columns:
        logger.info("Engineering features...")
        engineer = FeatureEngineer()
        df = engineer.create_features(df)
        feature_cols = engineer.get_feature_columns()
    else:
        # Assume features already exist
        feature_cols = [c for c in df.columns if c not in [
            'feed_timestamp', 'entity_id', 'trip_id', 'timestamp',
            'delay_sec', 'delay_min', 'delay_10plus', 'time_period'
        ]]
    
    logger.info(f"Using features: {feature_cols}")
    
    # Train
    trainer, metrics = train_from_dataframe(
        df,
        feature_cols=feature_cols,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nTest F1 Score: {metrics['test']['f1_score']:.4f}")
    print(f"Test Precision: {metrics['test']['precision']:.4f}")
    print(f"Test Recall: {metrics['test']['recall']:.4f}")
    print("\nTop 5 Important Features:")
    importance = trainer.get_feature_importance().head(5)
    for _, row in importance.iterrows():
        direction = "↑" if row['coefficient'] > 0 else "↓"
        print(f"  {direction} {row['feature']}: {row['coefficient']:.4f}")


if __name__ == '__main__':
    main()
