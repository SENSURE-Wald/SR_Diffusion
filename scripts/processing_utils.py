import rasterio
from tqdm import tqdm
import datetime
import numpy as np
from rasterio.warp import reproject
from rasterio.enums import Resampling
import rasterio
import warnings
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
import argparse
import os
import pathlib
import abc


class DOPProcessor(abc.ABC):

    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser()

        parser.add_argument("INPUT", type=pathlib.Path)
        parser.add_argument("OUTPUT", type=pathlib.Path)
        parser.add_argument("-s", "--size", default=800, type=int)

        self.args = parser.parse_args()

    @staticmethod
    def sanity_check(bounds, bbox, square_km):
        """
        Basic sanity checks which compare the data extracted from the file name with the data that is stated in the GeoTIF.
        """

        width = abs(bounds.right - bounds.left)
        height = abs(bounds.top - bounds.bottom)
        assert int(round(width)) == int(round(height)) == square_km * 1000
        assert int(round(bounds.left)) == int(round(bbox[0]))
        assert int(round(bounds.bottom)) == int(round(bbox[1]))
        assert int(round(bounds.right)) == int(round(bbox[2]))
        assert int(round(bounds.top)) == int(round(bbox[3]))

    @staticmethod
    def write_geotif(path, data, profile, date: datetime.datetime, size=800):

        scale_x = profile["width"] / size
        scale_y = profile["height"] / size
        new_transform = profile["transform"] * profile["transform"].scale(scale_x, scale_y)
        new_profile = profile.copy()
        new_profile.update(width=size, height=size, transform=new_transform)
        resampled = np.empty((data.shape[0], size, size), data.dtype)
        reproject(
            source=data,
            destination=resampled,
            src_transform=profile["transform"],
            src_crs=profile["crs"],
            dst_transform=new_transform,
            dst_crs=profile["crs"],
            resampling=Resampling.cubic,
        )

        with rasterio.open(path, "w", **new_profile) as dst:
            dst.write(resampled)
            dst.update_tags(recording_date=date.isoformat() if type(date) is datetime.datetime else date)

    @staticmethod
    def check_dict(left, bottom, date, path):
        """
        previous_path = None
        if (left, bottom) in existing_tiles:
            previous_tile = existing_tiles[(left, bottom)]
            previous_date = previous_tile["date"]
            previous_path = previous_tile["path"]
            if previous_date >= date:
                import sys

                print(path, previous_path)
                sys.exit(1)
                return False

            else:
                existing_tiles.update({(left, bottom): {"date": date, "path": path}})
                print(path, previous_path)
                try:
                    os.remove(previous_path)
                except TypeError as e:
                    pass
                return True
        """
        return True

    @staticmethod
    def tile_image_into_2x2(image):
        width, height = image.width, image.height
        half_w, half_h = width // 2, height // 2
        tiles = []
        windows = {
            "top_left": rasterio.windows.Window(0, 0, half_w, half_h),
            "top_right": rasterio.windows.Window(half_w, 0, width - half_w, half_h),
            "bottom_left": rasterio.windows.Window(0, half_h, half_w, height - half_h),
            "bottom_right": rasterio.windows.Window(half_w, half_h, width - half_w, height - half_h),
        }

        for name, window in windows.items():
            transform = image.window_transform(window)

            profile = image.profile.copy()
            profile.update(
                {
                    "height": window.height,
                    "width": window.width,
                    "transform": transform,
                }
            )
            tiles.append(
                {
                    "name": name,
                    "window": window,
                    "profile": profile,
                    "bounds": rasterio.windows.bounds(window, image.transform),
                }
            )

        return tiles

    @staticmethod
    def tile_image_into_grid(image, grid=(2, 2)):
        """Split a rasterio dataset into n_cols * n_rows windows.

        grid: (n_cols, n_rows), e.g. (5, 4) for a 5 km wide, 4 km high tile
        that should become twenty 1 km tiles.
        """
        n_cols, n_rows = grid
        if n_cols < 1 or n_rows < 1:
            raise ValueError(f"grid must be positive, got {grid}")
        if image.width % n_cols or image.height % n_rows:
            warnings.warn(
                f"{image.width}x{image.height} px is not divisible by grid {grid}; "
                "edge tiles will be slightly larger or smaller than the others"
            )

        # integer pixel boundaries; they cover the image exactly even if it is not divisible
        xs = [i * image.width // n_cols for i in range(n_cols + 1)]
        ys = [j * image.height // n_rows for j in range(n_rows + 1)]

        tiles = []
        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(xs[col], ys[row], xs[col + 1] - xs[col], ys[row + 1] - ys[row])

                profile = image.profile.copy()
                profile.update(
                    height=window.height,
                    width=window.width,
                    transform=image.window_transform(window),
                )

                tiles.append(
                    {
                        "name": f"r{row}_c{col}",
                        "row": row,
                        "col": col,
                        "window": window,
                        "profile": profile,
                        "bounds": window_bounds(window, image.transform),  # (left, bottom, right, top)
                    }
                )
        return tiles

    @staticmethod
    def get_bbox(coords: tuple[float], square_km=1):
        """
        Return the 1x1 km bounding box based on the coords provided in the file name (they are assumed to be still divided by 1000
        We are staying in the given coordinate frame from the source file.
        """
        # Min easting / northing
        min_e = coords[0] * 1000
        min_n = coords[1] * 1000
        max_e = min_e + square_km * 1000
        max_n = min_n + square_km * 1000

        return [min_e, min_n, max_e, max_n]

    @abc.abstractmethod
    def extract_from_file_name(self, file_name: str) -> tuple[tuple[float, float], datetime.date, int]:
        pass

    def process(self):

        os.makedirs(self.args.OUTPUT, exist_ok=True)

        """"
            We want only to keep the last recorded date of each tile. Therefore we will keep a running dictionary that is tagged like this:
            {
                [left, bottom]: {
                                    "date": date,
                                    "path": path
                                },
                ...
            }
            In the case a newer tile is found we will delete the old one and write the new one to disk. 
        """
        global existing_tiles
        existing_tiles = {}

        for file in tqdm(list(self.args.INPUT.rglob("tif/*.tif"))):
            coords, date, square_km = self.extract_from_file_name(file_name=file.name)
            if coords is None:
                continue
            bbox = self.get_bbox(coords, square_km=square_km)
            with rasterio.open(file, "r") as image:
                try:
                    self.sanity_check(image.bounds, bbox, square_km)
                except AssertionError as e:
                    print(file)
                    raise e
                if square_km == 1:
                    bounds = image.bounds
                    left, bottom = int(bounds[0] / 1000), int(bounds[1] / 1000)
                    output_path = self.args.OUTPUT / file.name

                    if output_path.exists():
                        continue

                    if self.check_dict(left, bottom, date, output_path):
                        data = image.read()
                        profile = image.profile.copy()
                        self.write_geotif(output_path, data, profile, date, self.args.size)

                else:
                    if square_km == 2:
                        square_km = (2, 2)

                    tiles = self.tile_image_into_grid(image, square_km)
                    for tile_dict in tiles:
                        profile = tile_dict["profile"]
                        bounds = tile_dict["bounds"]
                        window = tile_dict["window"]
                        # We need to bring our coords into format that is used in the file names.
                        new_left, new_bottom = int(bounds[0] / 1000), int(bounds[1] / 1000)

                        split_file_name = file.name.split("_")
                        prefix = split_file_name[0]
                        utm_zone = split_file_name[1]
                        date_str = split_file_name[-1].split(".")[0]

                        new_file_name = f"{prefix}_{utm_zone}_{new_left}_{new_bottom}_1_ni_{date_str}.tif"
                        output_path = self.args.OUTPUT / new_file_name
                        if output_path.exists():
                            continue

                        data = image.read(window=window)

                        if self.check_dict(new_left, new_bottom, date, output_path):
                            self.write_geotif(output_path, data, profile, date, self.args.size)
