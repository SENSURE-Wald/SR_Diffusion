"""Find tiles that overlap valid (non-nodata) pixels of a mask raster.

Usage:
  python3 filter_tiles.py MASK.tif DIR [DIR ...] --list     # write DIR_in_mask.txt per dir
  python3 filter_tiles.py MASK.tif DIR [DIR ...] --delete   # delete tiles outside mask (asks first)
"""
import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mask", type=Path, help="mask raster; its NoData value marks empty area")
p.add_argument("dirs", type=Path, nargs="+", help="tile folders (*.tif)")
p.add_argument("--list", action="store_true", help="write DIR_in_mask.txt with kept tile names")
p.add_argument("--delete", action="store_true", help="delete tiles without mask data, after y/N prompt")
args = p.parse_args()
if not (args.list or args.delete):
    p.error("pass --list and/or --delete")

mask_ds = gdal.Open(str(args.mask))
band = mask_ds.GetRasterBand(1)
nodata = band.GetNoDataValue()
if nodata is None:
    p.error(f"{args.mask} has no NoData value, cannot tell empty area")
mx0, mpx, _, my0, _, mpy = mask_ds.GetGeoTransform()


def has_data(tile):
    ds = gdal.Open(str(tile))
    x0, px, _, y0, _, py = ds.GetGeoTransform()
    x1, y1 = x0 + px * ds.RasterXSize, y0 + py * ds.RasterYSize
    # tile bounds -> mask pixel window, clipped to mask extent
    c0 = max(0, int(round((x0 - mx0) / mpx)))
    c1 = min(mask_ds.RasterXSize, int(round((x1 - mx0) / mpx)))
    r0 = max(0, int(round((y0 - my0) / mpy)))
    r1 = min(mask_ds.RasterYSize, int(round((y1 - my0) / mpy)))
    if c1 <= c0 or r1 <= r0:
        return False
    return bool(np.any(band.ReadAsArray(c0, r0, c1 - c0, r1 - r0) != nodata))


to_delete = []
for d in args.dirs:
    tiles = sorted(d.glob("*.tif"))
    kept, dropped = [], []
    for t in tiles:
        (kept if has_data(t) else dropped).append(t)
    print(f"{d}: {len(kept)}/{len(tiles)} tiles have mask data, {len(dropped)} without")
    if args.list:
        out = d.with_name(d.name + "_in_mask.txt")
        out.write_text("".join(t.name + "\n" for t in kept))
        print(f"  list -> {out}")
    to_delete += dropped

if args.delete and to_delete:
    print(f"\nWARNING: this permanently deletes {len(to_delete)} tile(s) outside the mask. No undo.")
    if input("Type y to continue: ").strip().lower() != "y":
        print("Aborted, nothing deleted.")
    else:
        for t in to_delete:
            t.unlink()
        print(f"Deleted {len(to_delete)} tile(s).")
