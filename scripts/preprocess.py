import argparse
from tqdm import tqdm
from pathlib import Path
import rasterio
import types
from opensr_model.data.DOP20Dataset import ImageDataset, MultiSpectralDataset, DOP20Dataset
from opensr_model.utils import save_to_tif
from scripts.processing_utils import DOPProcessor

parser = argparse.ArgumentParser()

parser.add_argument("INPUT", type=Path)
parser.add_argument("--size", "-s", default=400, help="Pixel width and height to downsample the images to.", type=int)
args = parser.parse_args()

image_size = (args.size, args.size)
lr_path = args.INPUT / "s2"
lr_dataset = MultiSpectralDataset(lr_path, (100, 100), list(range(13)), file_type="tif", gain=3, rgb_channels=[3, 2, 1])

hr_path = args.INPUT / "tif_800x800"
hr_dataset = ImageDataset(hr_path, image_size=(800, 800), channels=[0, 1, 2, 3], file_type="tif", gain=1)

harmonizer_config = types.SimpleNamespace()
harmonizer_config.name = "colouring"

paired_dataset = DOP20Dataset(lr_dataset, hr_dataset, harmonizer_config=harmonizer_config)

output_path = args.INPUT / "dop_h"

for i, (lr, hr) in tqdm(enumerate(paired_dataset), total=len(paired_dataset)):
    source_tif = paired_dataset.dataset_1.image_paths[i]

    if (output_path / source_tif.name).exists():
        continue

    with rasterio.open(source_tif, "r") as src:
        profile = src.profile
        profile.update({"dtype": rasterio.float32})

        date = src.tags()["recording_date"]
        DOPProcessor.write_geotif(output_path / source_tif.name, hr.cpu().numpy(), profile, date, args.size)
