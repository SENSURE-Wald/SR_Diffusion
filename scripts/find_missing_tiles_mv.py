import pandas as pd
import pathlib
import re

METADATA_PATH = pathlib.Path("/NAS/data/dop/MV/metadata")
TIF_PATH = pathlib.Path("/NAS/data/dop/MV/tif")

# --- Load metadata CSVs ---
dfs = []
for csv in METADATA_PATH.rglob("*.csv"):
    df = pd.read_csv(csv, skiprows=5, encoding="utf-8", sep=";")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Use the existing coordinate columns directly (no regex needed here)
combined["east"] = combined["Koordinatenursprung_East"].astype("Int64")
combined["north"] = combined["Koordinatenursprung_North"].astype("Int64")

csv_coords = set(zip(combined["east"], combined["north"]))

# --- Load tif filenames ---
pattern = re.compile(r"dop(\d+)rgbi_(\d+)_(\d+)_(\d+)_(\d+)_mv")

tif_records = []
for tif_file in TIF_PATH.rglob("*.tif"):
    stem = tif_file.name.split(".tif")[0]
    m = pattern.match(stem)
    if m is None:
        continue
    resolution, utm_zone, east_km, north_km, tile_idx = m.groups()
    tif_records.append(
        {
            "filename": tif_file.name,
            "east": int(east_km) * 1000,
            "north": int(north_km) * 1000,
        }
    )

tif_df = pd.DataFrame(tif_records)
tif_coords = set(zip(tif_df["east"], tif_df["north"]))

# --- Compare ---
missing_from_csv = tif_coords - csv_coords  # tiles that have a .tif but no metadata row
missing_from_tif = csv_coords - tif_coords  # tiles that have metadata but no .tif file

print(f"{len(missing_from_csv)} tiles have a .tif file but no matching CSV metadata entry")
print(f"{len(missing_from_tif)} tiles have CSV metadata but no matching .tif file")

# Get the actual filenames for the ones missing metadata
missing_files = tif_df[tif_df.apply(lambda r: (r["east"], r["north"]) in missing_from_csv, axis=1)]
print(missing_files[["filename", "east", "north"]])

# Get the actual Kachelname rows for coords with no tif
missing_meta = combined[combined.apply(lambda r: (r["east"], r["north"]) in missing_from_tif, axis=1)]
print(missing_meta[["Kachelname", "east", "north"]])

missing_files_out = missing_files[["filename", "east", "north"]].sort_values(["east", "north"])
missing_files_out.to_csv("/NAS/data/dop/MV/missing_tiles/missing_from_csv.csv", index=False)

missing_meta_out = missing_meta[["Kachelname", "east", "north"]].sort_values(["east", "north"])
missing_meta_out.to_csv("/NAS/data/dop/MV/missing_tiles/missing_from_tif.csv", index=False)

print(f"Saved {len(missing_files_out)} rows to missing_from_csv.csv")
print(f"Saved {len(missing_meta_out)} rows to missing_from_tif.csv")
