"""
TransLink GTFS-RT Data Collector

Collects real-time transit data from TransLink API and stores in GCS.
Can run as:
1. Local script: python -m src.data.collector
2. Cloud Function: deployed to GCP
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import requests
from google.cloud import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransLinkCollector:
    """Collector for TransLink GTFS-RT data."""
    
    # TransLink API endpoints
    TRIP_UPDATES_URL = "https://gtfs.translink.ca/v3/gtfstripupdates"
    VEHICLE_POSITIONS_URL = "https://gtfs.translink.ca/v3/gtfsposition"
    SERVICE_ALERTS_URL = "https://gtfs.translink.ca/v3/gtfsalerts"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        local_output_dir: Optional[Union[str, Path]] = None,
        use_gcs: bool = True
    ):
        """
        Initialize collector.
        
        Args:
            api_key: TransLink API key. If None, reads from TRANSLINK_API_KEY env var.
            bucket_name: GCS bucket name. If None, reads from GCS_RAW_BUCKET env var.
            local_output_dir: Local directory for output (if not using GCS).
            use_gcs: Whether to upload to GCS (False for local testing).
        """
        self.api_key = api_key or os.getenv('TRANSLINK_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set TRANSLINK_API_KEY env var.")
        
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name or os.getenv('GCS_RAW_BUCKET')
        
        if self.use_gcs:
            if not self.bucket_name:
                raise ValueError("GCS bucket required. Set GCS_RAW_BUCKET env var.")
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
        else:
            self.local_output_dir = Path(local_output_dir or 'data/raw/gtfs_rt/trip_updates')
            self.local_output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_trip_updates(self) -> bytes:
        """
        Fetch trip updates from TransLink API.
        
        Returns:
            Raw protobuf bytes.
            
        Raises:
            requests.RequestException: If API call fails.
        """
        url = f"{self.TRIP_UPDATES_URL}?apikey={self.api_key}"
        
        logger.info(f"Fetching trip updates from {self.TRIP_UPDATES_URL}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        logger.info(f"Fetched {len(response.content)} bytes")
        return response.content
    
    def fetch_vehicle_positions(self) -> bytes:
        """Fetch vehicle positions from TransLink API."""
        url = f"{self.VEHICLE_POSITIONS_URL}?apikey={self.api_key}"
        
        logger.info(f"Fetching vehicle positions from {self.VEHICLE_POSITIONS_URL}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        logger.info(f"Fetched {len(response.content)} bytes")
        return response.content
    
    def save_to_gcs(self, data: bytes, data_type: str = 'trip_updates') -> str:
        """
        Save data to Google Cloud Storage.
        
        Args:
            data: Raw bytes to save.
            data_type: Type of data (trip_updates, vehicle_positions, etc.)
            
        Returns:
            GCS URI of saved file.
        """
        timestamp = int(time.time())
        blob_name = f"{data_type}/{data_type}_{timestamp}.pb"
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(data)
        
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"Saved to {gcs_uri}")
        
        return gcs_uri
    
    def save_locally(self, data: bytes, data_type: str = 'trip_updates') -> Path:
        """
        Save data to local filesystem.
        
        Args:
            data: Raw bytes to save.
            data_type: Type of data.
            
        Returns:
            Path to saved file.
        """
        timestamp = int(time.time())
        output_path = self.local_output_dir / f"{data_type}_{timestamp}.pb"
        
        output_path.write_bytes(data)
        logger.info(f"Saved to {output_path}")
        
        return output_path
    
    def collect(self, data_type: str = 'trip_updates') -> Union[str, Path]:
        """
        Collect data from API and save.
        
        Args:
            data_type: 'trip_updates' or 'vehicle_positions'.
            
        Returns:
            Path/URI to saved data.
        """
        if data_type == 'trip_updates':
            data = self.fetch_trip_updates()
        elif data_type == 'vehicle_positions':
            data = self.fetch_vehicle_positions()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if self.use_gcs:
            return self.save_to_gcs(data, data_type)
        else:
            return self.save_locally(data, data_type)


# ============================================================================
# Cloud Function Entry Point
# ============================================================================

def collect_trip_updates_cloud_function(request):
    """
    Cloud Function entry point for scheduled collection.
    
    Triggered by Cloud Scheduler every 5 minutes.
    
    Args:
        request: Flask request object (unused but required by Cloud Functions).
        
    Returns:
        Tuple of (response_body, status_code).
    """
    try:
        collector = TransLinkCollector(use_gcs=True)
        gcs_uri = collector.collect('trip_updates')
        
        return {
            'status': 'success',
            'gcs_uri': gcs_uri,
            'timestamp': datetime.utcnow().isoformat()
        }, 200
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500


def collect_vehicle_positions_cloud_function(request):
    """Cloud Function entry point for vehicle positions."""
    try:
        collector = TransLinkCollector(use_gcs=True)
        gcs_uri = collector.collect('vehicle_positions')
        
        return {
            'status': 'success',
            'gcs_uri': gcs_uri,
            'timestamp': datetime.utcnow().isoformat()
        }, 200
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }, 500


# ============================================================================
# Local Execution
# ============================================================================

def main():
    """Run collection locally."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect TransLink GTFS-RT data')
    parser.add_argument(
        '--local', 
        action='store_true',
        help='Save locally instead of GCS'
    )
    parser.add_argument(
        '--type',
        choices=['trip_updates', 'vehicle_positions', 'both'],
        default='trip_updates',
        help='Type of data to collect'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/gtfs_rt/trip_updates',
        help='Local output directory (when --local)'
    )
    
    args = parser.parse_args()
    
    # Collect
    collector = TransLinkCollector(
        use_gcs=not args.local,
        local_output_dir=args.output_dir
    )
    
    if args.type in ['trip_updates', 'both']:
        result = collector.collect('trip_updates')
        print(f"Trip updates saved: {result}")
    
    if args.type in ['vehicle_positions', 'both']:
        result = collector.collect('vehicle_positions')
        print(f"Vehicle positions saved: {result}")
    
    print("Collection complete!")


if __name__ == '__main__':
    main()
