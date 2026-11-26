"""Cut a mask raster into pieces that match each tile exactly.

Usage:
  python3 cut-mask.py MASK.tif TILE_DIR OUT_DIR

For every *.tif in TILE_DIR (subfolders included) writes the same relative path
under OUT_DIR: the part of the mask
covering the same ground area as the tile (same bounds and CRS), at the mask's own
pixel size. Tile area outside the mask gets the mask's NoData value (0 if it has none).
"""
import argparse
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mask", type=Path, help="mask raster")
p.add_argument("tiles", type=Path, help="tile folder (*.tif)")
p.add_argument("out", type=Path, help="output folder for mask pieces")
args = p.parse_args()

# output files reuse tile names, so writing into the tile folder would overwrite the tiles
if args.out.resolve().is_relative_to(args.tiles.resolve()):
    p.error("OUT_DIR must not be TILE_DIR or inside it")

mask_ds = gdal.Open(str(args.mask))
_, mask_px, _, _, _, mask_py = mask_ds.GetGeoTransform()
mask_px, mask_py = abs(mask_px), abs(mask_py)
tiles = sorted(args.tiles.rglob("*.tif"))
if not tiles:
    p.error(f"no *.tif found in {args.tiles} or its subfolders")
for t in tiles:
    out = args.out / t.relative_to(args.tiles)  # keep subfolder structure (e.g. LS/, SA/)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.Open(str(t))
    x0, px, _, y0, _, py = ds.GetGeoTransform()
    x1, y1 = x0 + px * ds.RasterXSize, y0 + py * ds.RasterYSize
    gdal.Warp(
        str(out), mask_ds,
        outputBounds=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
        xRes=mask_px, yRes=mask_py,
        dstSRS=ds.GetProjection(), resampleAlg="near",
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )
print(f"Wrote {len(tiles)} mask piece(s) to {args.out}")
