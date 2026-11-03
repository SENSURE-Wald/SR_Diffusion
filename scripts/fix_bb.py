import pathlib
from pathlib import Path
from tqdm import tqdm
import re
import datetime
from bs4 import BeautifulSoup
import shutil

COORD_PATTERN = re.compile(r"(\d+)[-_](\d{3,6})")


def parse_flightdate(html: str):
    soup = BeautifulSoup(html, "html.parser")
    label = soup.find("td", string=lambda s: s and s.strip() == "Bildflugdatum:")
    if label is None:
        raise ValueError(f"The html does not include a flight date.")

    value = label.find_next_sibling("td")

    if value is None:
        raise ValueError(f"The html does not include a valid flight date.")

    date_str = value.get_text(strip=True)

    return date_str


def sanity_check(file_name_year, metadata_date: datetime.datetime) -> bool:

    return file_name_year == metadata_date.year


def parse_tile_coords(filename: str):
    """Extract (easting_km, northing_km) from a filename like
    'dop_33333-5880.tif' -> (33333, 5580). -> (33333 - 33000, 5580)"""
    match = COORD_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not find '<easting>-<northing>' pattern in filename: {filename}")
    easting_km, northing_km = match.groups()
    return int(easting_km) - 33000, int(northing_km)


if __name__ == "__main__":
    path = pathlib.Path("/NAS/data/dop/BB/tif")
    missing_tiles_path = pathlib.Path("/NAS/data/dop/BB/missing/")
    missing_tiles = []
    for file in tqdm(list(path.rglob("dop_*.tif"))):
        easting, northing = parse_tile_coords(file.name)
        meta_data_path = file.parent / file.name.replace(".tif", ".html")
        suffix = file.name.find(".tif")

        with open(meta_data_path, encoding="utf-8") as f:
            try:
                date = parse_flightdate(f.read())
                new_file_name = f"dop20_33_{easting}_{northing}_1_bb_{date}.tif"
                new_file_path = file.parent / new_file_name
                shutil.move(str(file), str(new_file_path))

            except ValueError as e:
                missing_tiles.append(file.name)
                shutil.move(str(file), missing_tiles_path / file.name)

    print(f"We are missing the metadata for {len(missing_tiles)} tiles.")
    with open(missing_tiles_path / "missing_tiles.txt", "w") as f:
        for tile in missing_tiles:
            f.write(tile + "\n")
