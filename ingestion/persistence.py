import os
import shutil
from config import CHROMA_PATH, BACKUP_ZIP


def save_backup():
    """
    Zip Chroma store from /tmp to Workspace.
    Call this after every ingest so the backup is always up to date.
    """
    backup_base = BACKUP_ZIP.replace(".zip", "")
    shutil.make_archive(backup_base, "zip", CHROMA_PATH)
    print(f"✓ Backup saved → {BACKUP_ZIP}")


def restore_backup():
    """
    Restore Chroma store from backup if the sqlite file doesn't exist in /tmp.
    Call this at the start of every script before touching Chroma.

    Three cases:
    1. sqlite file exists in /tmp   → already on disk, do nothing
    2. backup zip exists            → unzip to /tmp, ready to use
    3. no backup                    → fresh start, re-ingest required
    """
    db_file = os.path.join(CHROMA_PATH, "chroma.sqlite3")

    if os.path.exists(db_file):
        return  # already on disk, nothing to do

    if os.path.exists(BACKUP_ZIP):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        shutil.unpack_archive(BACKUP_ZIP, CHROMA_PATH)
        print(f"✓ Restored from {BACKUP_ZIP}")
    else:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        print("⚠ No backup found — starting fresh (re-ingest required)")