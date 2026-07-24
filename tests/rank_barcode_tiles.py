"""Rank tiles across all ROIs for the three reference-tile roles.

For each tile, reads channel `--ch` of the max projection across all barcode
rounds and derives:
  - min_std   : weakest-round std          -> ref_tile (registration robustness)
  - mean_std  : mean std across rounds      -> barcode_ref_tiles (rolony signal)
  - bright    : median of per-round p99.9    -> correction_tiles (bright rolonies)

Channel 0 is a proxy (barcode signal is spread across channels), but the ranking
is consistent across tiles. Standalone analysis helper, not a pytest test.

Usage:
    python tests/rank_barcode_tiles.py <data_path> [--prefix barcode_round]
            [--ch 0] [--border 2] [--nbarcode N] [--nref M]
"""
import argparse
import warnings

import numpy as np
from tifffile import TiffFile

warnings.filterwarnings("ignore")
from iss_preprocess.io import get_roi_dimensions  # noqa: E402
from iss_preprocess.io.load import get_processed_path  # noqa: E402

DEAD_STD = 5.0


def page_stats(path, ch):
    if not path.exists():
        return None
    with TiffFile(path) as t:
        a = t.pages[ch].asarray()
    return float(a.std()), float(np.percentile(a, 99.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("--prefix", default="barcode_round")
    ap.add_argument("--ch", type=int, default=0)
    ap.add_argument("--border", type=int, default=2)
    ap.add_argument("--nbarcode", type=int, default=20, help="# barcode_ref candidates")
    ap.add_argument("--ncorr", type=int, default=15, help="# correction candidates")
    ap.add_argument("--nref", type=int, default=10, help="# ref_tile candidates")
    args = ap.parse_args()

    pp = get_processed_path(args.data_path)
    nums = sorted(
        int(p.name.split("_")[2])
        for p in pp.glob(f"{args.prefix}_*_1")
        if p.is_dir() and p.name.endswith("_1")
    )
    rounds = [f"{args.prefix}_{n}_1" for n in nums]
    rd = get_roi_dimensions(args.data_path, "genes_round_1_1")
    print(f"{args.data_path} | ch{args.ch} | {len(rounds)} {args.prefix} | "
          f"{len(rd)} ROIs", flush=True)

    tiles = []  # dict per complete tile
    per_round_all = {r: [] for r in rounds}
    for roi, mx, my in rd:
        roi, nx, ny = int(roi), int(mx) + 1, int(my) + 1
        dead_acc = {r: [] for r in rounds}
        ncomplete = 0
        for ix in range(nx):
            for iy in range(ny):
                stds, p999 = {}, {}
                complete = True
                for r in rounds:
                    s = page_stats(
                        pp / r / f"{r}_MMStack_{roi}-Pos{ix:03d}_{iy:03d}_max.tif",
                        args.ch,
                    )
                    if s is None:
                        complete = False
                    else:
                        stds[r], p999[r] = s
                        dead_acc[r].append(s[0])
                        per_round_all[r].append(s[0])
                if complete and stds:
                    ncomplete += 1
                    sv = np.array(list(stds.values()))
                    pv = np.array(list(p999.values()))
                    tiles.append(dict(
                        roi=roi, x=ix, y=iy,
                        border=min(ix, nx - 1 - ix, iy, ny - 1 - iy),
                        min_std=float(sv.min()),
                        mean_std=float(sv.mean()),
                        bright=float(np.median(pv)),
                    ))
        dead = [r for r in rounds if dead_acc[r]
                and np.median(dead_acc[r]) < DEAD_STD]
        flag = f"  !! DEAD rounds: {dead}" if dead else ""
        print(f"  ROI {roi:2d}: {nx}x{ny}, {ncomplete} tiles complete{flag}", flush=True)

    print(f"\ntotal complete tiles across ROIs: {len(tiles)}", flush=True)

    def top(key, n, interior=False, spread=False):
        pool = [t for t in tiles if (t["border"] >= args.border or not interior)]
        pool = sorted(pool, key=lambda t: t[key], reverse=True)
        if not spread:
            return pool[:n]
        picked, seen = [], {}
        for t in pool:  # at most 2 per ROI, to spread coverage
            if seen.get(t["roi"], 0) < 2:
                picked.append(t); seen[t["roi"]] = seen.get(t["roi"], 0) + 1
            if len(picked) >= n:
                break
        return picked

    def fmt(t):
        return (f"[{t['roi']}, {t['x']}, {t['y']}]  min_std={t['min_std']:6.1f} "
                f"mean_std={t['mean_std']:6.1f} bright={t['bright']:7.1f} "
                f"border={t['border']}")

    print(f"\n===== ref_tile candidates (interior, by weakest-round std) =====")
    for t in top("min_std", args.nref, interior=True):
        print("  " + fmt(t))
    print(f"\n===== correction_tiles candidates (by brightness p99.9) =====")
    for t in top("bright", args.ncorr):
        print("  " + fmt(t))
    print(f"\n===== barcode_ref_tiles candidates (by mean signal, spread over ROIs) =====")
    picks = top("mean_std", args.nbarcode, spread=True)
    for t in picks:
        print("  " + fmt(t))
    print("\n  as a YAML list:")
    for t in picks:
        print(f"    - [{t['roi']}, {t['x']}, {t['y']}]")


if __name__ == "__main__":
    main()
