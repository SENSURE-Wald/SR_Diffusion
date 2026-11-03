from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from tqdm import tqdm


def _bbox_from_name(stem: str, tile_size=(1000, 1000)):
    """Parse '..._<easting_idx>_<northing_idx>_...' style filenames into a bbox."""
    coords = list(map(float, stem.split("_")[2:4]))
    min_e = tile_size[0] * coords[0]  # Easting,  bottom-left
    min_n = tile_size[1] * coords[1]  # Northing, bottom-left
    max_e = min_e + tile_size[0]
    max_n = min_n + tile_size[1]
    return min_e, min_n, max_e, max_n


def write_geotiff(file: Path, output_path: Path, epsg: int = 25832, tile_size=(1000, 1000)):
    """Convert a single image file to a georeferenced GeoTIFF."""
    min_e, min_n, max_e, max_n = _bbox_from_name(file.name, tile_size)

    output_path.mkdir(parents=True, exist_ok=True)
    output_name = output_path / file.with_suffix(".tif").name

    with rasterio.open(file) as src:
        data = src.read()
        count, height, width = data.shape

        pixel_width = (max_e - min_e) / width
        pixel_height = (max_n - min_n) / height

        transform = from_origin(min_e, max_n, pixel_width, pixel_height)

        profile = src.profile.copy()

        # JPEG-spezifische bzw. inkompatible Einstellungen entfernen
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("photometric", None)
        profile.pop("interleave", None)
        profile.update(
            driver="GTiff", transform=transform, crs=CRS.from_epsg(epsg), count=count, compress="None", tiles=False
        )

        with rasterio.open(output_name, "w", **profile) as dst:
            dst.write(data)

    print(f"Saved: {output_name}")


def write_geotiffs(folder: Path, output_path: Path, epsg: int = 25832, tile_size=(1000, 1000)):
    """Convert every image file in a folder to a georeferenced GeoTIFF."""
    output_path.mkdir(parents=True, exist_ok=True)

    for input_file in folder.iterdir():
        try:
            write_geotiff(input_file, output_path, epsg=epsg, tile_size=tile_size)
        except (IndexError, ValueError, rasterio.errors.RasterioIOError):
            pass


if __name__ == "__main__":
    input_path = Path("waldschaden")
    output_path = Path("waldschaden_geotiff/")
    epsg = 25832

    for file in tqdm(list(input_path.iterdir())):
        try:
            write_geotiff(file=file, output_path=output_path, epsg=epsg)
        except IndexError:
            pass
