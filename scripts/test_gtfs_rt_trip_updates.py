print("SCRIPT STARTED")

from pathlib import Path
import time
import requests
import os
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

load_dotenv()

API_KEY = os.getenv("TRANSLINK_API_KEY")
print("API_KEY loaded?", bool(API_KEY))

if not API_KEY:
    raise ValueError("TRANSLINK_API_KEY not found. Check your .env file at project root.")

url = f"https://gtfsapi.translink.ca/v3/gtfsrealtime?apikey={API_KEY}"
print("Requesting:", url)

raw_dir = Path("data/raw/gtfs_rt/trip_updates")
raw_dir.mkdir(parents=True, exist_ok=True)

ts = int(time.time())
raw_path = raw_dir / f"trip_updates_{ts}.pb"

response = requests.get(url, timeout=30)
print("HTTP status:", response.status_code)
response.raise_for_status()

raw_path.write_bytes(response.content)
print("Saved raw feed to:", raw_path)

feed = gtfs_realtime_pb2.FeedMessage()
feed.ParseFromString(response.content)

print("Feed timestamp:", feed.header.timestamp)
print("Number of entities:", len(feed.entity))

for e in feed.entity[:5]:
    print(
        "entity_id:", e.id,
        "| has_trip_update:", e.HasField("trip_update")
    )

print("SCRIPT FINISHED SUCCESSFULLY")