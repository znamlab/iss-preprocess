"""Scan tiles to pick a ref_tile / correction_tile with signal in every round.

For a dataset, reads one channel (default ref_ch=0) of every tile's max
projection across all genes+barcode rounds and reports, per ROI:
  - the median std per round (to spot dead/failed rounds),
  - the interior tiles ranked by their *weakest-round* std (a good ref_tile is
    strong in its weakest round).
Then it prints a chamber-wide best interior tile.

Standalone analysis helper (needs real processed data), not a pytest unit test.

Usage:
    python tests/scan_ref_tiles.py <data_path> [--roi N | --roi all]
                                   [--ch 0] [--border 2]
Example:
    python tests/scan_ref_tiles.py toksozi_in-vivo-BRISC/BRAC12106.1c/chamber_02
    python tests/scan_ref_tiles.py toksozi_in-vivo-BRISC/BRAC12106.1c/chamber_01 --roi 15
"""
import argparse
import warnings

import numpy as np
from tifffile import TiffFile

warnings.filterwarnings("ignore")
from iss_preprocess.io import get_roi_dimensions  # noqa: E402
from iss_preprocess.io.load import get_processed_path  # noqa: E402

DEAD_STD = 5.0  # per-round median std below this flags a failed/blank round


def discover_rounds(processed_path):
    """Return existing genes_round_*_1 then barcode_round_*_1 folders, in order."""
    rounds = []
    for fam in ("genes", "barcode"):
        nums = sorted(
            int(p.name.split("_")[2])
            for p in processed_path.glob(f"{fam}_round_*_1")
            if p.is_dir() and p.name.endswith("_1")
        )
        rounds += [f"{fam}_round_{n}_1" for n in nums]
    return rounds


def page_std(path, ch):
    if not path.exists():
        return None
    with TiffFile(path) as t:
        return float(t.pages[ch].asarray().std())


def scan_roi(pp, rounds, roi, nx, ny, ch, border_min):
    """Return (tiles, per_round) for one ROI.

    tiles: list of (ix, iy, weakest_std, weakest_round, border) for tiles present
        in every round. per_round: {round: [std over tiles]}.
    """
    per_round = {r: [] for r in rounds}
    tiles = []
    for ix in range(nx):
        for iy in range(ny):
            stds, complete = {}, True
            for r in rounds:
                f = pp / r / f"{r}_MMStack_{roi}-Pos{ix:03d}_{iy:03d}_max.tif"
                s = page_std(f, ch)
                if s is None:
                    complete = False
                else:
                    stds[r] = s
                    per_round[r].append(s)
            if complete and stds:
                wr = min(stds, key=stds.get)
                b = min(ix, nx - 1 - ix, iy, ny - 1 - iy)
                tiles.append((ix, iy, stds[wr], wr, b))
    tiles.sort(key=lambda t: t[2], reverse=True)
    return tiles, per_round


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path", help="relative processed dataset path")
    ap.add_argument("--roi", default="all", help="ROI id, or 'all' (default)")
    ap.add_argument("--ch", type=int, default=0, help="channel / ref_ch (default 0)")
    ap.add_argument(
        "--border",
        type=int,
        default=2,
        help="min tiles from any grid edge to count as interior (default 2)",
    )
    args = ap.parse_args()

    pp = get_processed_path(args.data_path)
    rounds = discover_rounds(pp)
    rd = get_roi_dimensions(args.data_path, "genes_round_1_1")
    if args.roi != "all":
        rd = rd[rd[:, 0] == int(args.roi)]
    print(f"dataset {args.data_path} | ch{args.ch} | {len(rounds)} rounds | "
          f"{len(rd)} ROI(s)", flush=True)

    chamber_best = None  # (weakest_std, roi, ix, iy, weakest_round, border)
    for roi, mx, my in rd:
        roi, nx, ny = int(roi), int(mx) + 1, int(my) + 1
        tiles, per_round = scan_roi(pp, rounds, roi, nx, ny, args.ch, args.border)
        print(f"\n========== ROI {roi}  (grid {nx} x {ny}) ==========", flush=True)
        dead = [r for r in rounds if per_round[r]
                and np.median(per_round[r]) < DEAD_STD]
        if dead:
            print(f"  !! dead/blank rounds (median std < {DEAD_STD}): {dead}")
        interior = [t for t in tiles if t[4] >= args.border]
        print(f"  {len(tiles)} tiles complete in all rounds, "
              f"{len(interior)} interior (border>={args.border})")
        for ix, iy, ms, wr, b in interior[:5]:
            print(f"    [{roi}, {ix}, {iy}]  weakest std {ms:6.1f}  ({wr})  border {b}")
        if interior:
            ix, iy, ms, wr, b = interior[0]
            if chamber_best is None or ms > chamber_best[0]:
                chamber_best = (ms, roi, ix, iy, wr, b)

    if chamber_best:
        ms, roi, ix, iy, wr, b = chamber_best
        print(f"\n>>> CHAMBER-WIDE best interior ref_tile: [{roi}, {ix}, {iy}]  "
              f"(weakest round {wr}, std {ms:.1f}, border {b})")


if __name__ == "__main__":
    main()
