import csv
from tqdm import tqdm
import requests
import pathlib
import argparse
import os
import re
from urllib.parse import unquote


def main():
    # Matches "file=" followed by everything up to the next "&" or end of string
    FILE_PARAM_RE = re.compile(r"file=([^&]+)")
    with open(args.PATH, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        next(reader)

        for line in tqdm(reader, total=row_count):
            link = line["rgbi"]
            file_name = link.split("/")[-1]
            if not file_name.endswith(".tif"):
                match = FILE_PARAM_RE.search(file_name)
                file_name = unquote(match.group(1))

            output_path: pathlib.Path = store_path / file_name

            if output_path.is_file():
                continue

            try:
                response = requests.get(link)

                with open(output_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.ChunkedEncodingError as e:
                main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path)

    args = parser.parse_args()

    store_path = args.output if args.output is not None else args.PATH.parent / "tif"

    os.makedirs(store_path, exist_ok=True)

    with open(args.PATH, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        row_count = sum(1 for _ in reader)

    main()
