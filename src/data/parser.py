"""
GTFS-RT Protobuf Parser

Parses raw Protocol Buffer files from TransLink into structured CSV/Parquet.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from google.transit import gtfs_realtime_pb2
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GTFSRTParser:
    """Parser for GTFS-Realtime protobuf files."""
    
    def __init__(self, use_gcs: bool = False, bucket_name: Optional[str] = None):
        """
        Initialize parser.
        
        Args:
            use_gcs: Whether to read from/write to GCS.
            bucket_name: GCS bucket name (required if use_gcs=True).
        """
        self.use_gcs = use_gcs
        
        if use_gcs:
            if not bucket_name:
                raise ValueError("bucket_name required when use_gcs=True")
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
    
    def read_pb_file(self, file_path: Union[str, Path]) -> gtfs_realtime_pb2.FeedMessage:
        """
        Read a protobuf file.
        
        Args:
            file_path: Local path or GCS URI (gs://bucket/path).
            
        Returns:
            Parsed FeedMessage.
        """
        file_path_str = str(file_path)
        
        if file_path_str.startswith('gs://'):
            # Read from GCS
            blob_name = file_path_str.replace(f'gs://{self.bucket.name}/', '')
            blob = self.bucket.blob(blob_name)
            content = blob.download_as_bytes()
            logger.info(f"Read {len(content)} bytes from GCS: {file_path_str}")
        else:
            # Read locally
            content = Path(file_path).read_bytes()
            logger.info(f"Read {len(content)} bytes from local: {file_path}")
        
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(content)
        
        return feed
    
    def parse_trip_updates(
        self, 
        feed: gtfs_realtime_pb2.FeedMessage
    ) -> pd.DataFrame:
        """
        Parse trip updates from feed.
        
        Args:
            feed: Parsed FeedMessage.
            
        Returns:
            DataFrame with trip update data.
        """
        feed_timestamp = feed.header.timestamp
        rows = []
        
        for entity in feed.entity:
            if not entity.HasField('trip_update'):
                continue
            
            tu = entity.trip_update
            trip = tu.trip
            
            # Trip-level info
            trip_id = getattr(trip, 'trip_id', None)
            route_id = getattr(trip, 'route_id', None)
            direction_id = getattr(trip, 'direction_id', None)
            start_date = getattr(trip, 'start_date', None)
            start_time = getattr(trip, 'start_time', None)
            
            # Stop-level updates
            for stu in tu.stop_time_update:
                stop_id = getattr(stu, 'stop_id', None)
                stop_sequence = getattr(stu, 'stop_sequence', None)
                
                # Arrival info
                arr_delay = None
                arr_time = None
                if stu.HasField('arrival'):
                    if stu.arrival.HasField('delay'):
                        arr_delay = stu.arrival.delay
                    if stu.arrival.HasField('time'):
                        arr_time = stu.arrival.time
                
                # Departure info
                dep_delay = None
                dep_time = None
                if stu.HasField('departure'):
                    if stu.departure.HasField('delay'):
                        dep_delay = stu.departure.delay
                    if stu.departure.HasField('time'):
                        dep_time = stu.departure.time
                
                # Use arrival delay if available, else departure delay
                delay_sec = arr_delay if arr_delay is not None else dep_delay
                
                rows.append({
                    'feed_timestamp': feed_timestamp,
                    'entity_id': entity.id,
                    'trip_id': trip_id,
                    'route_id': route_id,
                    'direction_id': direction_id,
                    'start_date': start_date,
                    'start_time': start_time,
                    'stop_id': stop_id,
                    'stop_sequence': stop_sequence,
                    'delay_sec': delay_sec,
                    'delay_min': delay_sec / 60.0 if delay_sec is not None else None,
                    'arrival_time_epoch': arr_time,
                    'departure_time_epoch': dep_time,
                })
        
        df = pd.DataFrame(rows)
        
        # Create outcome variable
        if 'delay_min' in df.columns:
            df['delay_10plus'] = df['delay_min'] >= 10.0
        
        logger.info(f"Parsed {len(df)} stop-time updates from {len(feed.entity)} entities")
        
        return df
    
    def parse_vehicle_positions(
        self,
        feed: gtfs_realtime_pb2.FeedMessage
    ) -> pd.DataFrame:
        """
        Parse vehicle positions from feed.
        
        Args:
            feed: Parsed FeedMessage.
            
        Returns:
            DataFrame with vehicle position data.
        """
        feed_timestamp = feed.header.timestamp
        rows = []
        
        for entity in feed.entity:
            if not entity.HasField('vehicle'):
                continue
            
            v = entity.vehicle
            
            rows.append({
                'feed_timestamp': feed_timestamp,
                'entity_id': entity.id,
                'trip_id': getattr(v.trip, 'trip_id', None),
                'route_id': getattr(v.trip, 'route_id', None),
                'vehicle_id': getattr(v.vehicle, 'id', None),
                'latitude': v.position.latitude if v.HasField('position') else None,
                'longitude': v.position.longitude if v.HasField('position') else None,
                'bearing': v.position.bearing if v.HasField('position') else None,
                'speed': v.position.speed if v.HasField('position') else None,
                'current_stop_sequence': v.current_stop_sequence if v.HasField('current_stop_sequence') else None,
                'current_status': v.current_status if v.HasField('current_status') else None,
                'timestamp': v.timestamp if v.HasField('timestamp') else None,
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"Parsed {len(df)} vehicle positions")
        
        return df
    
    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        data_type: str = 'trip_updates'
    ) -> pd.DataFrame:
        """
        Parse a single file and optionally save.
        
        Args:
            input_path: Path to .pb file (local or GCS).
            output_path: Where to save CSV (optional).
            data_type: 'trip_updates' or 'vehicle_positions'.
            
        Returns:
            Parsed DataFrame.
        """
        # Parse
        feed = self.read_pb_file(input_path)
        
        if data_type == 'trip_updates':
            df = self.parse_trip_updates(feed)
        elif data_type == 'vehicle_positions':
            df = self.parse_vehicle_positions(feed)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
            
            logger.info(f"Saved to {output_path}")
        
        return df
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = '*.pb'
    ) -> List[Path]:
        """
        Process all files in a directory.
        
        Args:
            input_dir: Directory with .pb files.
            output_dir: Where to save CSVs.
            pattern: Glob pattern for files.
            
        Returns:
            List of output file paths.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        for pb_file in sorted(input_dir.glob(pattern)):
            logger.info(f"Processing {pb_file}")
            
            # Determine output path
            timestamp = pb_file.stem.split('_')[-1]
            output_path = output_dir / f"trip_updates_parsed_{timestamp}.csv"
            
            # Skip if already exists
            if output_path.exists():
                logger.info(f"Skipping {output_path} (already exists)")
                continue
            
            try:
                df = self.process_file(pb_file, output_path)
                output_files.append(output_path)
                logger.info(f"Processed {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to process {pb_file}: {e}")
        
        logger.info(f"Processed {len(output_files)} files")
        return output_files


def main():
    """Run parser locally."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse GTFS-RT protobuf files')
    parser.add_argument(
        'input',
        type=str,
        help='Input file or directory'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file or directory'
    )
    parser.add_argument(
        '--type',
        choices=['trip_updates', 'vehicle_positions'],
        default='trip_updates',
        help='Type of data'
    )
    parser.add_argument(
        '--gcs',
        action='store_true',
        help='Read from GCS'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        help='GCS bucket name'
    )
    
    args = parser.parse_args()
    
    # Initialize parser
    parser_obj = GTFSRTParser(use_gcs=args.gcs, bucket_name=args.bucket)
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Process directory
        if not args.output:
            raise ValueError("--output required when input is directory")
        output_files = parser_obj.process_directory(
            input_path,
            args.output
        )
        print(f"Processed {len(output_files)} files to {args.output}")
    else:
        # Process single file
        df = parser_obj.process_file(
            input_path,
            args.output,
            data_type=args.type
        )
        print(f"Processed {len(df)} rows")
        print(df.head())


if __name__ == '__main__':
    main()
