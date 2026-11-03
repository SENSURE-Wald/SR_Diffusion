import pandas as pd
import pathlib
import re
import shutil
from tqdm import tqdm

METADATA_PATH = pathlib.Path("/NAS/data/dop/MV/metadata")
TIF_PATH = pathlib.Path("/NAS/data/dop/MV/tif")
MISSING_PATH = pathlib.Path("/NAS/data/dop/MV/missing_tiles")

DRY_RUN = False  # <-- set to False once you've checked the printed plan

if not DRY_RUN:
    MISSING_PATH.mkdir(parents=True, exist_ok=True)

# --- Load metadata CSVs ---
dfs = []
for csv in METADATA_PATH.rglob("*.csv"):
    df = pd.read_csv(csv, skiprows=5, encoding="utf-8", sep=";")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined["east"] = combined["Koordinatenursprung_East"].astype("Int64")
combined["north"] = combined["Koordinatenursprung_North"].astype("Int64")
combined["Aktualitaet"] = pd.to_datetime(combined["Aktualitaet"], format="%d.%m.%Y").dt.strftime("%Y-%m-%d")

date_lookup = dict(zip(zip(combined["east"], combined["north"]), combined["Aktualitaet"]))

# --- Patterns ---
# Not yet renamed: dop20rgbi_33_452_5946_2_mv
pattern_raw = re.compile(r"^dop(\d+)rgbi_(\d+)_(\d+)_(\d+)_(\d+)_mv$")
# Already renamed: dop20rgbi_33_452_5946_2_mv_2024-05-02
pattern_done = re.compile(r"^dop(\d+)rgbi_(\d+)_(\d+)_(\d+)_(\d+)_mv_\d{4}-\d{2}-\d{2}$")

renamed = []
moved_missing = []
skipped_already_done = []
errors = []

for tif_file in tqdm(list(TIF_PATH.rglob("*.tif"))):
    stem = tif_file.stem

    if pattern_done.match(stem):
        skipped_already_done.append(tif_file.name)
        continue

    m = pattern_raw.match(stem)
    if m is None:
        errors.append((tif_file.name, "filename did not match expected pattern"))
        continue

    resolution, utm_zone, east_km, north_km, tile_idx = m.groups()
    east = int(east_km) * 1000
    north = int(north_km) * 1000

    befliegungsdatum = date_lookup.get((east, north))

    if befliegungsdatum is None:
        dest = MISSING_PATH / tif_file.name
        if DRY_RUN:
            print(f"[DRY RUN] would move {tif_file.name} -> {dest}")
        else:
            shutil.move(str(tif_file), str(dest))
        moved_missing.append(tif_file.name)
        continue

    new_name = f"dop{resolution}rgbi_{utm_zone}_{east_km}_{north_km}_{tile_idx}_mv_{befliegungsdatum}.tif"
    new_path = tif_file.with_name(new_name)

    if DRY_RUN:
        print(f"[DRY RUN] would rename {tif_file.name} -> {new_name}")
    else:
        tif_file.rename(new_path)
    renamed.append((tif_file.name, new_name))

print(f"\n{'[DRY RUN] ' if DRY_RUN else ''}Renamed: {len(renamed)}")
print(f"{'[DRY RUN] ' if DRY_RUN else ''}Moved to missing_tiles: {len(moved_missing)}")
print(f"Skipped (already renamed): {len(skipped_already_done)}")
if errors:
    print(f"{len(errors)} files did not match any expected pattern:")
    for name, reason in errors:
        print(f"  {name}: {reason}")

prefix = "dry_run_" if DRY_RUN else ""
pd.DataFrame(renamed, columns=["old_name", "new_name"]).to_csv(
    f"/NAS/data/dop/MV/logs/{prefix}rename_log.csv", index=False
)
pd.DataFrame(moved_missing, columns=["moved_to_missing"]).to_csv(
    f"/NAS/data/dop/MV/logs/{prefix}moved_missing_log.csv", index=False
)
