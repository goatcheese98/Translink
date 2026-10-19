## Setup

### 1) Clone the repo

```sh
git clone https://github.com/goatcheese98/Translink.git
cd Translink
```

### 2) Create and activate a virtual environment

```sh
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```sh
source .venv/bin/activate
```

### 3) Install dependencies for the feed scripts

This repository does not include a `requirements.txt`. The two scripts below import these third-party packages:

```sh
python -m pip install requests python-dotenv gtfs-realtime-bindings pandas
```

These dependencies cover the feed scripts only; scientific analyses may require additional packages.

### 4) Configure the environment variable

`scripts/test_gtfs_rt_trip_updates.py` reads `TRANSLINK_API_KEY` from the environment and calls `load_dotenv()`. For a future authorized live download, create a local `.env` file at the repository root with a key issued to you:

```dotenv
TRANSLINK_API_KEY=<your-translink-api-key>
```

Replace the placeholder locally. `.env` is ignored by Git. Keep real credentials out of documentation, commits, and shared logs.

### 5) Run scripts

Parse the latest saved `trip_updates_*.pb` snapshot without a live request:

```sh
python scripts/parse_trip_updates_to_csv.py
```

This writes a CSV under `data/interim/gtfs_rt`. The repository includes saved snapshots under `data/raw/gtfs_rt/trip_updates`.

For an explicitly authorized live download, `python scripts/test_gtfs_rt_trip_updates.py` uses the configured key and writes a new snapshot. **The current download script prints the request URL, including the API key.** Treat its output as sensitive and do not capture or share it in logs.

### Security note

Apparent credentials were removed from the current README, but remain in Git history. This documentation cleanup does not revoke credentials or erase history. **Rohan must rotate or revoke anything still valid**, including any reused credentials. No credentials were tested as part of this cleanup.
