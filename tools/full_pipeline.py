"""
Runs the full Discord resource pipeline: fetch_messages, embed_store, batch_detect, repo_sync.
Logs all key steps and statistics to a log file for auditing and debugging.
"""
import subprocess
import logging
import os
from datetime import datetime

LOG_PATH = "tools/full_pipeline.log"
# Reset the log file at the start of each run
with open(LOG_PATH, "w"):
    pass
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

def run_and_log(cmd, step_name):
    logging.info(f"START: {step_name}")
    print(f"Running: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip()
        logging.info(line)
        print(line)
    proc.wait()
    if proc.returncode == 0:
        logging.info(f"SUCCESS: {step_name}")
        print(f"SUCCESS: {step_name}")
    else:
        logging.error(f"FAIL: {step_name} (exit code {proc.returncode})")
        print(f"FAIL: {step_name} (exit code {proc.returncode})")
    return proc.returncode

def main():
    steps = [
        ("python3 core/fetch_messages.py", "Fetch Discord messages"),
        ("python3 core/embed_store.py", "Embed and store messages"),
        ("python3 core/batch_detect.py", "Detect and enrich resources"),
        ("python3 core/repo_sync.py", "Export resources to JSON/Markdown"),
    ]
    for cmd, name in steps:
        rc = run_and_log(cmd, name)
        if rc != 0:
            print(f"Step failed: {name}. See {LOG_PATH} for details.")
            break
    # Log summary stats
    try:
        import sqlite3
        db_path = "data/discord_messages.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM messages")
        n_msgs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM resources")
        n_resources = cur.fetchone()[0]
        msg = f"Total messages in DB: {n_msgs} | Total resources in DB: {n_resources}"
        logging.info(msg)
        print(msg)
        conn.close()
    except Exception as e:
        logging.error(f"Failed to log DB stats: {e}")
        print(f"Failed to log DB stats: {e}")

if __name__ == "__main__":
    main()
