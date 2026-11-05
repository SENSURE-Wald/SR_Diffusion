import rasterio
import pathlib
from tqdm import tqdm
import numpy as np
from datetime import datetime

INPUT = pathlib.Path("/NAS/data/dop/BY/")
RGB_INPUT = INPUT / "rgb"
CIR_INPUT = INPUT / "cir"
OUTPUT = INPUT / "tif"


def check_rgb_cir(rgb: np.array, cir: np.array) -> bool:
    r, g = rgb[0, :, :], rgb[1, :, :]
    r_, g_ = cir[1, :, :], cir[2, :, :]

    mask = cir[3, :, :] / 255

    equal = np.array_equal(r[mask == 1], r_[mask == 1]) and np.array_equal(g[mask == 1], g_[mask == 1])

    """
    if not equal:
        d_r = np.abs(r, r_)
        d_g = np.abs(g, g_)

        print(d_r.max(), d_r.mean())
        print(d_g.max(), d_g.mean(), np.any(1 - mask))
    """

    return equal


broken_tifs = []
missing_tifs = []
all_tifs = list(RGB_INPUT.rglob("*.tif"))
for tif_file in tqdm(all_tifs):
    northing, easting = map(lambda x: int(x), tif_file.stem.split("_"))
    northing -= 32000
    dummy_output = f"dop20_32_{northing}_{easting}_1_by_"
    if len(list(OUTPUT.rglob(f"{dummy_output}*"))) != 0:
        continue

    with rasterio.open(tif_file, "r") as rgb_tif:
        rgb = rgb_tif.read()
        date_str = rgb_tif.tags()["BILDFLUG_DATUM"]
        date = datetime.strptime(date_str, "%d.%m.%Y")
        output_file = OUTPUT / f"dop20_32_{northing}_{easting}_1_by_{date.strftime("%Y-%m-%d")}.tif"
        cir_file_name = CIR_INPUT / f"{northing * 1000}_{easting * 1000}_cir.tif"
        try:
            with rasterio.open(cir_file_name, "r") as cir_tif:
                cir = cir_tif.read()

                if not check_rgb_cir(rgb, cir):
                    broken_tifs.append((str(tif_file), str(cir_file_name)))
                    print(
                        f"{len(broken_tifs)} of {len(all_tifs)} have a discprecancy between the images. This is {round((len(broken_tifs) / len(all_tifs)) * 100, 4)}%"
                    )

                profile = rgb_tif.profile.copy()
                profile.update(
                    {
                        "count": profile["count"] + 1,
                    }
                )
                with rasterio.open(output_file, "w+", **profile) as dst:
                    dst.write(rgb, (1, 2, 3))
                    dst.write(cir[0, :, :], 4)
        except rasterio.errors.RasterioIOError as e:
            print(f"For ({northing}, {easting}) file {tif_file}, no corresponding CIR file {cir_file_name} was found.")
            missing_tifs.append(cir_file_name)
            continue

print(
    f"{len(broken_tifs)} of {len(all_tifs)} have a discprecancy between the images. This is {round((len(broken_tifs) / len(all_tifs)) * 100, 4)}%"
)
with open(INPUT / "broken.txt", "w") as f:
    for line in broken_tifs:
        f.write(f"{line}\n")

with open(INPUT / "missing.txt", "w") as f:
    for line in missing_tifs:
        f.write(f"{line}\n")
