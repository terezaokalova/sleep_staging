import os, re

from pathlib import Path
print(os.environ["HOME"])
BASE = Path(os.environ["HOME"])/"Documents/STAT4830/STAT-4830-GOALZ-project/data"
# BASE = "/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/data"

PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"

# a simple regex to pull out IDs like "SC4032E0" or "ST7062J0"
id_pat = re.compile(r'^(SC|ST)\d{4}[A-Z]\d')

# collect raw IDs
raw_files = os.listdir(PROCESSED_DATA_DIR)
raw_ids = {
    m.group(0)
    for f in raw_files
    if (m := id_pat.match(f))
}

# collect catch22 IDs
c22_files = os.listdir(CATCH22_DATA_DIR)
c22_ids = {
    m.group(0)
    for f in c22_files
    if (m := id_pat.match(f))
}

print(f"{len(raw_ids)} raw IDs:", sorted(raw_ids))
print(f"{len(c22_ids)} catch-22 IDs:", sorted(c22_ids))
print("intersection:", sorted(raw_ids & c22_ids))
