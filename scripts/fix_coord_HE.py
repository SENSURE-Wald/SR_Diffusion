import pathlib
from pathlib import Path
import rasterio
from tqdm import tqdm
import re
import pandas as pd
import datetime
from rasterio.transform import from_origin

# Matches e.g. "434-5557" or "434_5557" anywhere in the filename
COORD_PATTERN = re.compile(r"(\d{3,6})[-_](\d{3,6})")
CRS = "EPSG:25832"


def sanity_check(file_name_year, metadata_date: datetime.datetime) -> bool:

    return file_name_year == metadata_date.year


def georeference_file(src_path: Path, dst_path: Path, recording_date, tile_size_m: float = 1000.0):
    easting_km, northing_km = parse_tile_coords(src_path.name)
    min_x = easting_km * 1000
    min_y = northing_km * 1000
    max_y = min_y + tile_size_m

    with rasterio.open(src_path) as src:
        width = src.width
        height = src.height
        data = src.read()
        profile = src.profile.copy()

    pixel_size_x = tile_size_m / width
    pixel_size_y = tile_size_m / height
    transform = from_origin(min_x, max_y, pixel_size_x, pixel_size_y)

    profile.update(
        crs=CRS,
        transform=transform,
        driver="GTiff",
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(recording_date=recording_date.isoformat())


def parse_tile_coords(filename: str):
    """Extract (easting_km, northing_km) from a filename like
    'TDOP_434-5557_BF2023.tif' -> (434, 5557)."""
    match = COORD_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not find '<easting>-<northing>' pattern in filename: {filename}")
    easting_km, northing_km = match.groups()
    return int(easting_km), int(northing_km)


path = pathlib.Path("/NAS/data/dop/HE/raw")
df = pd.read_excel(path.parent / "Metadaten_DOP_Oktober-2025.xlsx", header=0)
tile_names = df["Kachelname"]
recording_dates = df["Bildflugdatum"]
missing_tiles = []
tiles_by_dates = {}
old_tiles = []
for file in tqdm(list(path.rglob("*.tif"))):
    easting, northing = parse_tile_coords(file.name)
    suffix = file.name.find(".tif")
    year = int(file.name[suffix - 4 : suffix])
    search_str = f"DOP20_32_{easting}_{northing}_1_he"

    index = tile_names[tile_names == search_str].first_valid_index()
    try:
        recording_date_pd = recording_dates[index]
        recording_date = recording_date_pd.to_pydatetime()
        if search_str not in tiles_by_dates:
            tiles_by_dates[search_str] = [recording_date.isoformat()]

        else:
            tiles_by_dates[search_str].append(recording_date.isoformat())

        if not sanity_check(year, recording_date):
            raise ValueError(
                f"The year {year} from the file name {file.name} does not match the metadata date {recording_date_pd}"
            )

        output_filename = search_str.replace("DOP", "dop")
        output_filename = f"{output_filename}_{recording_date.strftime("%Y-%m-%d")}.tif"
        output_path = Path("/NAS/data/dop/HE/tif") / output_filename
        if output_path.exists():
            continue
        georeference_file(file, output_path, recording_date=recording_date)

    except KeyError:
        missing_tiles.append(search_str)
        continue

    except ValueError:
        old_tiles.append(search_str)
        continue


print(f"There are {len(missing_tiles)} missing tiles.")
print(f"There are {len(old_tiles)} old tiles.")

with open(path.parent / "old_tiles.txt", "w") as f:
    for tile in old_tiles:
        f.write(tile + "\n")

with open(path.parent / "missing_tiles.txt", "w") as f:
    for tile in missing_tiles:
        f.write(tile + "\n")

for entries in tiles_by_dates.values():
    if len(entries) != 1:
        print(entries)
# georeference_file(
#    file,
#    output_path,
# )
