from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from blade.types import PairResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_id TEXT NOT NULL,
  ref_id TEXT NOT NULL,
  sharpness_ok INTEGER NOT NULL,
  inlier_ratio REAL,
  blade_width_px REAL,
  area_ratio_2017 REAL,
  area_ratio_2018 REAL,
  area_delta REAL,
  tier TEXT NOT NULL,
  reason TEXT NOT NULL,
  thresholds_used TEXT NOT NULL,
  overlay_uri TEXT,
  num_matches INTEGER NOT NULL DEFAULT 0,
  num_inliers INTEGER NOT NULL DEFAULT 0,
  lap_2018 REAL,
  lap_2017 REAL,
  sharp_2018_ok INTEGER,
  sharp_2017_ok INTEGER,
  pipeline_furthest TEXT,
  pipeline_steps TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(query_id, ref_id)
);

CREATE TABLE IF NOT EXISTS overrides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pair_id INTEGER NOT NULL,
  old_tier TEXT NOT NULL,
  new_tier TEXT NOT NULL,
  note TEXT,
  at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(pair_id) REFERENCES pairs(id)
);

CREATE INDEX IF NOT EXISTS idx_pairs_inlier ON pairs(inlier_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_pairs_tier ON pairs(tier);
"""

_EXTRA_COLS = {
    "lap_2018": "REAL",
    "lap_2017": "REAL",
    "sharp_2018_ok": "INTEGER",
    "sharp_2017_ok": "INTEGER",
    "pipeline_furthest": "TEXT",
    "pipeline_steps": "TEXT",
}


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("thresholds_used", "pipeline_steps"):
        raw = d.get(key)
        if isinstance(raw, str) and raw:
            try:
                d[key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return d


class ResultStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        existing = {
            r[1] for r in self._conn.execute("PRAGMA table_info(pairs)").fetchall()
        }
        for col, typ in _EXTRA_COLS.items():
            if col not in existing:
                self._conn.execute(f"ALTER TABLE pairs ADD COLUMN {col} {typ}")

    def upsert_pair(self, result: PairResult) -> int:
        row = result.to_row()
        cur = self._conn.execute(
            """
            INSERT INTO pairs (
              query_id, ref_id, sharpness_ok, inlier_ratio, blade_width_px,
              area_ratio_2017, area_ratio_2018, area_delta, tier, reason,
              thresholds_used, overlay_uri, num_matches, num_inliers,
              lap_2018, lap_2017, sharp_2018_ok, sharp_2017_ok,
              pipeline_furthest, pipeline_steps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query_id, ref_id) DO UPDATE SET
              sharpness_ok=excluded.sharpness_ok,
              inlier_ratio=excluded.inlier_ratio,
              blade_width_px=excluded.blade_width_px,
              area_ratio_2017=excluded.area_ratio_2017,
              area_ratio_2018=excluded.area_ratio_2018,
              area_delta=excluded.area_delta,
              tier=excluded.tier,
              reason=excluded.reason,
              thresholds_used=excluded.thresholds_used,
              overlay_uri=excluded.overlay_uri,
              num_matches=excluded.num_matches,
              num_inliers=excluded.num_inliers,
              lap_2018=excluded.lap_2018,
              lap_2017=excluded.lap_2017,
              sharp_2018_ok=excluded.sharp_2018_ok,
              sharp_2017_ok=excluded.sharp_2017_ok,
              pipeline_furthest=excluded.pipeline_furthest,
              pipeline_steps=excluded.pipeline_steps,
              created_at=datetime('now')
            """,
            (
                row["query_id"],
                row["ref_id"],
                int(row["sharpness_ok"]),
                row["inlier_ratio"],
                row["blade_width_px"],
                row["area_ratio_2017"],
                row["area_ratio_2018"],
                row["area_delta"],
                row["tier"],
                row["reason"],
                json.dumps(row["thresholds_used"]),
                row["overlay_uri"],
                row["num_matches"],
                row["num_inliers"],
                row["lap_2018"],
                row["lap_2017"],
                None if row["sharp_2018_ok"] is None else int(row["sharp_2018_ok"]),
                None if row["sharp_2017_ok"] is None else int(row["sharp_2017_ok"]),
                row["pipeline_furthest"],
                json.dumps(row["pipeline_steps"]),
            ),
        )
        self._conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        got = self._conn.execute(
            "SELECT id FROM pairs WHERE query_id=? AND ref_id=?",
            (row["query_id"], row["ref_id"]),
        ).fetchone()
        return int(got["id"])

    def list_pairs(
        self,
        *,
        limit: int = 100,
        tier: str | None = None,
        min_inlier: float | None = None,
        query_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM pairs WHERE 1=1"
        args: list[Any] = []
        if tier:
            sql += " AND tier=?"
            args.append(tier)
        if query_id:
            sql += " AND query_id=?"
            args.append(query_id)
        if min_inlier is not None:
            sql += " AND inlier_ratio IS NOT NULL AND inlier_ratio>=?"
            args.append(min_inlier)
        sql += " ORDER BY inlier_ratio IS NULL, inlier_ratio DESC, id DESC LIMIT ?"
        args.append(limit)
        return [_row_to_dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def get_pair(self, pair_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM pairs WHERE id=?", (pair_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def best_per_query(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT p.* FROM pairs p
            INNER JOIN (
              SELECT query_id, MAX(inlier_ratio) AS best
              FROM pairs
              WHERE inlier_ratio IS NOT NULL
              GROUP BY query_id
            ) t ON p.query_id = t.query_id AND p.inlier_ratio = t.best
            ORDER BY p.inlier_ratio DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
