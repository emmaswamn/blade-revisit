#!/usr/bin/env python3
"""Background land-boundary layout lines — eyeball + pair scores.

Writes data/debug_preview/bg_layout_*.jpg
See docs/background-match.md (v3 layout lines).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.background import (  # noqa: E402
    compose_bg_layout_pair_debug,
    draw_bg_layout_debug,
    extract_bg_layout,
    layout_result_to_sig,
    score_bg_layout,
    search_bg_layout_gallery,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="bg land-boundary layout probe")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "wind-turbine-dtu")
    ap.add_argument("--search", action="store_true", help="also run full usable gallery search")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / "data" / "debug_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    query = ("2018", "DJI_0190.JPG")
    refs = [
        ("2017", "DJI_0752.JPG"),
        ("2017", "DJI_0593.JPG"),
        ("2017", "DJI_0765.JPG"),
        ("2017", "DJI_0713.JPG"),
    ]

    qy, qn = query
    qb = cv2.imread(str(args.data_root / qy / qn))
    if qb is None:
        raise FileNotFoundError(args.data_root / qy / qn)
    q_res = extract_bg_layout(qn, qb, year=qy)
    q_sig = layout_result_to_sig(q_res)
    q_vis = draw_bg_layout_debug(q_res)
    q_out = out_dir / f"bg_layout_{Path(qn).stem}.jpg"
    cv2.imwrite(str(q_out), q_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"{qy}/{qn} segs={len(q_res.segments)} ink={(q_res.ink > 0).mean():.4f} -> {q_out}")

    pair_rows: list[dict] = []
    for ry, rn in refs:
        path = args.data_root / ry / rn
        rb = cv2.imread(str(path))
        if rb is None:
            print(f"missing {path}")
            continue
        r_res = extract_bg_layout(rn, rb, year=ry)
        r_sig = layout_result_to_sig(r_res)
        vote = score_bg_layout(q_sig, r_sig)
        panel = compose_bg_layout_pair_debug(q_res, r_res, score=vote["score"])
        pout = out_dir / f"bg_layout_{Path(qn).stem}_vs_{Path(rn).stem}.jpg"
        cv2.imwrite(str(pout), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        r_vis = draw_bg_layout_debug(r_res)
        cv2.imwrite(
            str(out_dir / f"bg_layout_{Path(rn).stem}.jpg"),
            r_vis,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        print(
            f"  vs {rn}: score={vote['score']:.3f} "
            f"chamfer={vote['chamfer']} ori={vote['ori']} band={vote['band']} -> {pout.name}"
        )
        pair_rows.append({"ref": f"{ry}/{rn}", **vote, "preview": pout.name})

    meta = {
        "query": f"{qy}/{qn}",
        "pairs": pair_rows,
    }
    if args.search:
        summary = search_bg_layout_gallery(
            data_root=args.data_root,
            query_name=qn,
            query_year=qy,
            top_k=12,
            limit=args.limit,
            preview_dir=out_dir,
            log_print=True,
        )
        meta["search"] = {
            "gold_rank": summary.get("gold_rank"),
            "secs": summary.get("secs"),
            "top": summary.get("top"),
            "method": summary.get("method"),
        }
        print(f"\ngold_rank={summary.get('gold_rank')} secs={summary.get('secs')}")
        for row in summary["top"]:
            mark = " ← GOLD" if row["name"] == summary.get("gold_name") else ""
            print(
                f"  #{row['rank']} {row['name']} score={row['score']:.3f} "
                f"ch={row['chamfer']} ori={row['ori']}{mark}"
            )

    meta_path = out_dir / "bg_layout_0190_probe.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
