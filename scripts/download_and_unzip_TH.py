from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathlib
import zipfile
import argparse
import requests
import json
import io

parser = argparse.ArgumentParser()
parser.add_argument("INPUT", type=pathlib.Path)
parser.add_argument("OUTPUT", type=pathlib.Path)
parser.add_argument("-t", "--threads", type=int, default=8, help="Number of worker threads")

args = parser.parse_args()

with open(args.INPUT, "r") as f:
    links_dict = json.load(f)


def download_and_extract(link, output_dir):
    link = link.strip()
    file_name = link.split("/")[-1]
    tif_file_name = file_name.replace(".zip", ".tif")
    if (args.OUTPUT / tif_file_name).exists():
        return

    response = requests.get(link)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_file:
        zip_file.extractall(output_dir)

    return link


args.OUTPUT.mkdir(parents=True, exist_ok=True)

links = list(links_dict.values())
errors = []

with ThreadPoolExecutor(max_workers=args.threads) as executor:

    futures = {executor.submit(download_and_extract, link, args.OUTPUT): link for link in links}

    for future in tqdm(as_completed(futures), total=len(futures)):
        link = futures[future]

        try:
            future.result()
        except Exception as e:
            errors.append((link, str(e)))

if errors:
    print(f"\n{len(errors)} download(s) failed:")
    for link, err in errors:
        print(f"  {link}: {err}")
