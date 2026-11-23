"""Run database: every build is recorded, you rate it, and the ratings feed the learning loop.

    python -m forge.runs list
    python -m forge.runs rate 7 good
    python -m forge.runs rate 8 bad --tags baked_shadows,wrong_scale --note "door too narrow"
    python -m forge.runs export ratings.csv        # training data for the reward model

Failure tags (keep them consistent, they become labels): warped_facade, floating_props,
baked_shadows, wrong_scale, texture_seams, not_on_style, sign_text_garbled, too_clean,
black_interior, walk_through_walls, invisible_in_game, flicker_at_distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created REAL, spec_id TEXT, spec_hash TEXT, spec_json TEXT,
  summary_json TEXT, texture_mb REAL, validation_json TEXT, passed INTEGER,
  resource TEXT, seconds REAL,
  rating TEXT, tags TEXT, note TEXT, rated REAL
);
"""


def _db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def record_run(path: Path, spec: dict, spec_hash: str, summary: dict, textures: dict, report: dict,
               resource: str, seconds: float) -> int:
    con = _db(path)
    cur = con.execute(
        "INSERT INTO runs (created, spec_id, spec_hash, spec_json, summary_json, texture_mb, validation_json, passed, "
        "resource, seconds) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (time.time(), spec["id"], spec_hash, json.dumps(spec), json.dumps(summary),
         sum(t["bytes"] for t in textures.values()) / 1e6, json.dumps(report),
         None if report.get("passed") is None else int(bool(report["passed"])), resource, seconds))
    con.commit()
    return cur.lastrowid


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m forge.runs")
    ap.add_argument("--db", default="build/runs.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("rate")
    r.add_argument("run_id", type=int)
    r.add_argument("rating", choices=["good", "bad", "1", "2", "3", "4", "5"])
    r.add_argument("--tags", default="")
    r.add_argument("--note", default="")
    e = sub.add_parser("export")
    e.add_argument("csv")
    a = ap.parse_args(argv)
    con = _db(Path(a.db))
    if a.cmd == "list":
        for row in con.execute("SELECT id, datetime(created,'unixepoch'), spec_id, passed, rating, tags FROM runs ORDER BY id"):
            print(*row, sep="  ")
    elif a.cmd == "rate":
        n = con.execute("UPDATE runs SET rating=?, tags=?, note=?, rated=? WHERE id=?",
                        (a.rating, a.tags, a.note, time.time(), a.run_id)).rowcount
        con.commit()
        print("rated" if n else f"no run {a.run_id}")
    elif a.cmd == "export":
        rows = con.execute("SELECT id, spec_id, spec_hash, summary_json, texture_mb, passed, rating, tags, note FROM runs "
                           "WHERE rating IS NOT NULL").fetchall()
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "spec_id", "spec_hash", "summary_json", "texture_mb", "passed", "rating", "tags", "note"])
            w.writerows(rows)
        print(f"{len(rows)} rated runs → {a.csv}")


if __name__ == "__main__":
    main()
