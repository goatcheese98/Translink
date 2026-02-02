"""
Feature Engineering for Transit Delay Prediction

Creates ML features from parsed GTFS-RT data.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for delay prediction."""
    
    def __init__(self, rolling_window_days: int = 28):
        """
        Initialize feature engineer.
        
        Args:
            rolling_window_days: Days for rolling statistics.
        """
        self.rolling_window_days = rolling_window_days
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with 'feed_timestamp' column.
            
        Returns:
            DataFrame with added temporal features.
        """
        df = df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['feed_timestamp'], unit='s')
        
        # Extract components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rush hour flags
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_evening_rush'] = df['hour'].isin([16, 17, 18]).astype(int)
        
        # Time period buckets
        def get_time_period(hour):
            if 6 <= hour < 9:
                return 'morning_rush'
            elif 9 <= hour < 12:
                return 'midday'
            elif 12 <= hour < 16:
                return 'afternoon'
            elif 16 <= hour < 19:
                return 'evening_rush'
            elif 19 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        df['time_period'] = df['hour'].apply(get_time_period)
        
        logger.info("Added temporal features")
        return df
    
    def compute_route_stats(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute route-level statistics.
        
        Args:
            df: DataFrame to add features to.
            reference_df: DataFrame to compute stats from (for avoiding leakage).
                         If None, uses df.
                         
        Returns:
            DataFrame with route stats.
        """
        df = df.copy()
        ref = reference_df if reference_df is not None else df
        
        # Route statistics
        route_stats = ref.groupby('route_id').agg({
            'delay_min': ['mean', 'std', 'median', 'count'],
            'delay_10plus': 'mean'
        }).reset_index()
        
        route_stats.columns = [
            'route_id',
            'route_avg_delay',
            'route_std_delay',
            'route_median_delay',
            'route_trip_count',
            'route_delay_rate'
        ]
        
        # Fill NaN std with 0 (routes with only one observation)
        route_stats['route_std_delay'] = route_stats['route_std_delay'].fillna(0)
        
        # Merge
        df = df.merge(route_stats, on='route_id', how='left')
        
        logger.info(f"Computed stats for {len(route_stats)} routes")
        return df
    
    def compute_stop_stats(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute stop-level statistics.
        
        Args:
            df: DataFrame to add features to.
            reference_df: DataFrame to compute stats from.
            
        Returns:
            DataFrame with stop stats.
        """
        df = df.copy()
        ref = reference_df if reference_df is not None else df
        
        # Stop statistics
        stop_stats = ref.groupby('stop_id').agg({
            'delay_min': ['mean', 'median', 'count'],
            'delay_10plus': 'mean'
        }).reset_index()
        
        stop_stats.columns = [
            'stop_id',
            'stop_avg_delay',
            'stop_median_delay',
            'stop_trip_count',
            'stop_delay_rate'
        ]
        
        # Merge
        df = df.merge(stop_stats, on='stop_id', how='left')
        
        logger.info(f"Computed stats for {len(stop_stats)} stops")
        return df
    
    def compute_route_stop_stats(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute route+stop combination statistics.
        
        This is often the most predictive feature.
        """
        df = df.copy()
        ref = reference_df if reference_df is not None else df
        
        # Route-stop statistics
        rs_stats = ref.groupby(['route_id', 'stop_id']).agg({
            'delay_min': 'mean',
            'delay_10plus': 'mean',
            'stop_sequence': 'mean'
        }).reset_index()
        
        rs_stats.columns = [
            'route_id',
            'stop_id',
            'route_stop_avg_delay',
            'route_stop_delay_rate',
            'avg_stop_sequence'
        ]
        
        # Merge
        df = df.merge(rs_stats, on=['route_id', 'stop_id'], how='left')
        
        logger.info(f"Computed stats for {len(rs_stats)} route-stop pairs")
        return df
    
    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with global defaults.
        
        Important: This should be done AFTER train/test split to avoid leakage.
        """
        df = df.copy()
        
        # Global defaults
        global_delay_rate = df['delay_10plus'].mean()
        global_avg_delay = df['delay_min'].mean()
        
        # Fill route stats
        df['route_delay_rate'] = df['route_delay_rate'].fillna(global_delay_rate)
        df['route_avg_delay'] = df['route_avg_delay'].fillna(global_avg_delay)
        df['route_std_delay'] = df['route_std_delay'].fillna(0)
        df['route_trip_count'] = df['route_trip_count'].fillna(0)
        
        # Fill stop stats
        df['stop_delay_rate'] = df['stop_delay_rate'].fillna(global_delay_rate)
        df['stop_avg_delay'] = df['stop_avg_delay'].fillna(global_avg_delay)
        df['stop_trip_count'] = df['stop_trip_count'].fillna(0)
        
        # Fill route-stop stats with route stats as fallback
        df['route_stop_delay_rate'] = df['route_stop_delay_rate'].fillna(df['route_delay_rate'])
        df['route_stop_avg_delay'] = df['route_stop_avg_delay'].fillna(df['route_avg_delay'])
        
        # Fill stop_sequence with median
        df['stop_sequence'] = df['stop_sequence'].fillna(df['stop_sequence'].median())
        
        logger.info("Filled missing values")
        return df
    
    def create_features(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: Input DataFrame.
            reference_df: Reference data for statistics (to avoid leakage).
            fill_missing: Whether to fill missing values.
            
        Returns:
            DataFrame with all features.
        """
        df = df.copy()
        
        # Temporal features
        df = self.add_temporal_features(df)
        
        # Historical stats
        df = self.compute_route_stats(df, reference_df)
        df = self.compute_stop_stats(df, reference_df)
        df = self.compute_route_stop_stats(df, reference_df)
        
        # Fill missing if requested
        if fill_missing:
            df = self.fill_missing(df)
        
        logger.info(f"Created features. Shape: {df.shape}")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names for modeling."""
        return [
            'hour',
            'day_of_week',
            'is_weekend',
            'is_rush_hour',
            'stop_sequence',
            'route_avg_delay',
            'route_std_delay',
            'route_delay_rate',
            'stop_avg_delay',
            'stop_delay_rate',
            'route_stop_delay_rate',
            'route_stop_avg_delay',
        ]
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        Prepare data for training with proper train/test split.
        
        Args:
            df: Input DataFrame.
            test_size: Fraction for test set.
            random_state: Random seed.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_cols).
        """
        from sklearn.model_selection import train_test_split
        
        # Sort by time to avoid leakage
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Split index
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Compute stats on train only
        train_df = self.create_features(train_df, reference_df=train_df)
        test_df = self.create_features(test_df, reference_df=train_df)
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        
        # Check all features exist
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        # Prepare X, y
        X_train = train_df[feature_cols].values
        y_train = train_df['delay_10plus'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['delay_10plus'].values
        
        return X_train, X_test, y_train, y_test, feature_cols


def main():
    """Run feature engineering locally."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Engineer features from parsed data')
    parser.add_argument(
        'input',
        type=str,
        help='Input CSV file or directory'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='Output CSV file'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=28,
        help='Rolling window days for stats'
    )
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if input_path.is_dir():
        # Load all CSVs
        csv_files = list(input_path.glob('*.csv'))
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(csv_files)} files, {len(df)} rows")
    else:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")
    
    # Engineer features
    engineer = FeatureEngineer(rolling_window_days=args.window)
    df_features = engineer.create_features(df)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    
    logger.info(f"Saved features to {output_path}")
    print(f"Features: {engineer.get_feature_columns()}")
    print(f"Shape: {df_features.shape}")


if __name__ == '__main__':
    main()
