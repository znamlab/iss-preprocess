import warnings, numpy as np
warnings.filterwarnings("ignore")
from tifffile import TiffFile
from iss_preprocess.io import get_roi_dimensions
from iss_preprocess.io.load import get_processed_path

dp = "toksozi_in-vivo-BRISC/BRAC12106.1c/chamber_01"
ROI, CH = 15, 0
pp = get_processed_path(dp)
rounds = [f"genes_round_{i}_1" for i in range(1, 8)] + \
         [f"barcode_round_{i}_1" for i in range(1, 15)]

rd = get_roi_dimensions(dp, "genes_round_1_1")
row = rd[rd[:, 0] == ROI][0]
nx, ny = int(row[1]) + 1, int(row[2]) + 1
print(f"ROI {ROI}: grid {nx} x {ny} (x:0-{nx-1}, y:0-{ny-1})", flush=True)


def page_std(path):
    if not path.exists():
        return None
    with TiffFile(path) as t:
        return float(t.pages[CH].asarray().std())


tiles = []
for ix in range(nx):
    for iy in range(ny):
        stds = {}
        complete = True
        for r in rounds:
            s = page_std(pp / r / f"{r}_MMStack_{ROI}-Pos{ix:03d}_{iy:03d}_max.tif")
            if s is None:
                complete = False
            else:
                stds[r] = s
        if complete and stds:
            wr = min(stds, key=stds.get)
            # border distance: how many tiles from the nearest grid edge
            border = min(ix, nx - 1 - ix, iy, ny - 1 - iy)
            tiles.append((ix, iy, stds[wr], wr, border))

tiles.sort(key=lambda t: t[2], reverse=True)
print(f"\n{len(tiles)} tiles present in all rounds. (border = #tiles from nearest grid edge)")
print(f"{'tile':>14}  {'weakest_std':>11}  {'weakest_round':>18}  border")
for ix, iy, ms, wr, b in tiles:
    print(f"  [{ROI}, {ix}, {iy}]  {ms:11.1f}  {wr:>18}  {b}")

print("\n=== best INTERIOR tiles (border >= 2, i.e. >=2 tiles in from any edge) ===")
interior = [t for t in tiles if t[4] >= 2]
for ix, iy, ms, wr, b in interior[:8]:
    print(f"  [{ROI}, {ix}, {iy}]  weakest std {ms:6.1f}  ({wr})  border {b}")
if interior:
    ix, iy, ms, wr, b = interior[0]
    print(f"\nRECOMMENDED interior ref_tile: [{ROI}, {ix}, {iy}]  (weakest round {wr}, std {ms:.1f})")
