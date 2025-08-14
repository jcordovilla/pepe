#!/usr/bin/env python3
import sqlite3
from datetime import datetime
from pathlib import Path

db_path = Path('data/resources.db')
if not db_path.exists():
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
c = conn.cursor()
rows = c.execute('SELECT id, timestamp FROM resources').fetchall()
updated = 0
for row in rows:
    id_, ts = row
    if ts:
        try:
            new_ts = datetime.fromisoformat(ts).strftime('%Y-%m-%d')
        except Exception:
            new_ts = ts[:10]
        if new_ts != ts:
            c.execute('UPDATE resources SET timestamp = ? WHERE id = ?', (new_ts, id_))
            updated += 1
conn.commit()
conn.close()
print(f"Timestamps updated to YYYY-MM-DD for {updated} rows.") 