# Automated Data Collection Plan

## Overview

This document outlines the plan for automated, non-local data collection of TransLink GTFS Realtime data over a 1-week period to support our transit delay prediction model.

**Last Updated:** 2026-02-02  
**Collection Period:** 7 consecutive days  
**Primary Metric:** Precision (delay prediction reliability)

---

## 1. Platform Selection

### Chosen Platform: Google Cloud Platform (GCP) Compute Engine f1-micro

| Platform | Free Tier | Storage | 5-Min Viable? | Why Chosen/Rejected |
|----------|-----------|---------|---------------|---------------------|
| **GCP f1-micro** ✅ | Forever | 30 GB | Yes | Sufficient storage, always-free, persistent disk |
| PythonAnywhere Free | Forever | ~512 MB | No | Storage limit exceeded at 5-min intervals |
| PythonAnywhere Paid | $5/month | 10 GB | Maybe | Good alternative if GCP setup is too complex |
| GitHub Actions | 2000 min/mo | None | No | Minutes exhausted quickly at 5-min frequency |
| AWS EC2 t2.micro | 12 months | EBS | Yes | Only free for 12 months, then costs $ |

**Rationale:** GCP f1-micro provides the best balance of free-tier longevity (forever), adequate storage (30 GB), and acceptable performance for simple API polling tasks.

---

## 2. Data Collection Strategy

### Frequency: Every 5 Minutes

**Why 5 minutes?**
- Bus delays don't change meaningfully at 30-second or 1-minute intervals
- Provides sufficient temporal diversity for modeling
- Manageable storage (~2-4 GB for 1 week)
- Captures rush hour transitions without over-collection

### Operating Hours: 5:00 AM - 1:00 AM

**Why not 24/7?**
- TransLink regular bus service: ~5:00 AM - 12:30 AM (weekdays)
- NightBus (N-routes) operate 1:30 AM - 5:00 AM but target different user segment
- 20-hour window captures commuter behavior (early workers to late commuters)
- ~20% storage savings vs 24-hour collection

**Cron Schedule:**
```cron
# Weekdays (Mon-Fri) + Weekends, 5 AM - 12:55 AM
*/5 5-23 * * * cd ~/transit-delay-prediction && python3 scripts/collect_data.py
*/5 0-1 * * * cd ~/transit-delay-prediction && python3 scripts/collect_data.py
```

---

## 3. Storage Estimates

| Metric | Value |
|--------|-------|
| Snapshots per day | 240 (20 hours × 12 per hour) |
| Snapshots per week | ~1,680 |
| Avg file size (raw .pb) | ~50-150 KB |
| Avg file size (parsed CSV) | ~100-300 KB |
| Total storage (raw only) | ~1-2 GB |
| Total storage (raw + CSV) | ~2-4 GB |
| GCP free limit | 30 GB |
| **Headroom** | **~26-28 GB** |

---

## 4. Setup Instructions

### Prerequisites
- Google account
- Credit card (required for GCP verification, won't be charged on free tier)
- GitHub repo access

### Step 1: GCP Account Setup (~10 min)
1. Navigate to [cloud.google.com](https://cloud.google.com)
2. Sign up with Google account
3. Enter credit card for verification
4. Complete phone verification
5. Accept terms of service

### Step 2: Enable Compute Engine (~5 min)
1. Go to GCP Console → Navigation menu → Compute Engine
2. Click "Enable Compute Engine API" (takes 1-2 minutes)

### Step 3: Create f1-micro Instance (~10 min)
1. Go to Compute Engine → VM Instances
2. Click "Create Instance"
3. Configure:
   - **Name:** `translink-collector`
   - **Region:** `us-west1` (Oregon) - closest to Vancouver
   - **Machine type:** `f1-micro` (1 vCPU, 0.6 GB memory)
   - **Boot disk:** Ubuntu 22.04 LTS, 30 GB standard persistent disk
   - **Firewall:** Check "Allow HTTP traffic" (optional)
4. Click "Create"

### Step 4: SSH and Environment Setup (~20 min)
1. Click "SSH" button next to your instance (opens browser terminal)
2. Run setup commands:

```bash
# Update packages
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone repository
git clone https://github.com/yourusername/transit-delay-prediction.git
cd transit-delay-prediction

# Install dependencies
pip3 install -r requirements.txt

# Create environment file
echo "TRANSLINK_API_KEY=your_api_key_here" > .env
```

3. Test the collection script:
```bash
python3 scripts/collect_data.py
```

### Step 5: Configure Cron (~5 min)
1. Edit crontab:
```bash
crontab -e
```

2. Add collection schedule:
```cron
# TransLink data collection - every 5 minutes, 5 AM to 1 AM daily
*/5 5-23 * * * cd ~/transit-delay-prediction && python3 scripts/collect_data.py >> ~/collector.log 2>&1
*/5 0-1 * * * cd ~/transit-delay-prediction && python3 scripts/collect_data.py >> ~/collector.log 2>&1
```

3. Verify cron is set:
```bash
crontab -l
```

### Step 6: Set Up Billing Alerts (~5 min)
1. Go to Billing → Budgets & alerts
2. Create budget:
   - **Amount:** $0.01 (triggers alert before any charges)
   - **Alert threshold:** 50%, 90%, 100%
   - **Email recipients:** Your email

---

## 5. Monitoring & Maintenance

### Check Collection Status
```bash
# View recent logs
tail -f ~/collector.log

# Check disk usage
df -h

# Count collected files
ls -1 data/raw/gtfs_rt/trip_updates/ | wc -l
```

### Common Issues

| Issue | Solution |
|-------|----------|
| API key errors | Check .env file, verify key with TransLink |
| Disk full (unlikely) | Delete old .pb files, keep only CSV |
| Script crashes | Check logs, ensure dependencies installed |
| Instance stopped | Restart via GCP Console, cron resumes automatically |

### Weekly Checklist
- [ ] Verify disk usage < 50% (target: stay under 15 GB)
- [ ] Check collector.log for errors
- [ ] Confirm ~240 files/day being created
- [ ] Download data backup if critical (optional)

---

## 6. Data Collection Script Modifications

Ensure `scripts/collect_data.py` exists and handles:
- API failures gracefully (retry logic)
- Timestamp-based file naming to avoid overwrites
- Logging to stdout (captured by cron)

Example skeleton:
```python
import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    timestamp = int(time.time())
    output_dir = Path("data/raw/gtfs_rt/trip_updates")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Your API call logic here
    logger.info(f"Collected data at {timestamp}")

if __name__ == "__main__":
    main()
```

---

## 7. Post-Collection: Data Retrieval

### Download Data to Local Machine
```bash
# From your local terminal
gcloud compute scp translink-collector:~/transit-delay-prediction/data/ ./local-backup/ --recurse
```

### Or Keep on Cloud for Processing
- Continue model training directly on GCP instance
- Upload to Google Cloud Storage for sharing
- Push to GitHub (if < 100 MB, use LFS otherwise)

---

## 8. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| GCP free tier changes | Low | Monitor email alerts, have PythonAnywhere fallback |
| Disk fills up | Low | 30 GB limit, 5-min schedule, 20-hour window = plenty of headroom |
| TransLink API downtime | Medium | Script should fail gracefully, cron retries in 5 min |
| Instance crashes | Low | GCP auto-restart, cron resumes on boot |
| Lost SSH access | Low | Use browser-based SSH as fallback |

---

## 9. Alternative: Simplified PythonAnywhere Approach

If GCP setup proves too complex, use **PythonAnywhere paid ($5/month)**:

1. Sign up at pythonanywhere.com
2. Upload project files via web interface
3. Open Bash console, install requirements
4. Create "Always-on task":
   ```bash
   cd ~/transit-delay-prediction && python3 run_collector_loop.py
   ```
   Where `run_collector_loop.py` contains:
   ```python
   import time
   from scripts.collect_data import main
   
   while True:
       main()
       time.sleep(300)  # 5 minutes
   ```

**Note:** 10 GB storage limit may require deleting old data mid-week.

---

## Summary

| Parameter | Value |
|-----------|-------|
| **Platform** | GCP Compute Engine f1-micro |
| **Cost** | Free (forever tier) |
| **Frequency** | Every 5 minutes |
| **Hours** | 5:00 AM - 1:00 AM (20 hours/day) |
| **Duration** | 7 days |
| **Expected Storage** | 2-4 GB |
| **Expected Snapshots** | ~1,680 |
| **Setup Time** | ~50 minutes |
| **Monitoring** | Weekly log checks, billing alerts |

---

## Appendix: Quick Command Reference

```bash
# SSH to instance
gcloud compute ssh translink-collector

# View running processes
ps aux | grep python

# Check cron logs
grep CRON /var/log/syslog

# Restart cron
sudo service cron restart

# Check API key is set
cat .env

# Manual test run
python3 scripts/collect_data.py

# Disk cleanup (if needed)
find data/raw/gtfs_rt/trip_updates/ -name "*.pb" -mtime +7 -delete
```
