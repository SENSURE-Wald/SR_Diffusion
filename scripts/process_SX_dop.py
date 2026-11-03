import datetime
import csv

from scripts.processing_utils import DOPProcessor


class SaxonyProcessor(DOPProcessor):

    # Saxony includes around 11% images that were recorded on multiple days and stitched together.
    # These 11% will be excluded from the collection right now.
    def extract_from_file_name(self, file_name: str) -> tuple[tuple[float, float], datetime.date, int]:
        """
        An example file name would be dop20rgbi_33278_5590_2_sn.tif
        From this we want to extract:
            - the coords 278, 5590 (they are divided by 1000)
            - the date is not included in the filename but can be extracted from the metadata file dop20rgbi_33278_5590_2_sn_akt.csv
            - the size of the tile which should always be 2x2 km.
        """
        split_file_name = file_name.split("_")
        coords = list(map(float, split_file_name[1:3]))

        coords[0] -= 33000

        # Now we need to read the metadata CSV file. They look like this:
        # Kachelname;Ausdehnung (xmin ymin xmax ymax);Aktualitaet
        # 2785590;278000 5590000 280000 5592000;2025-04-03
        metadata_name = file_name.replace(".tif", "_akt.csv")
        metadata_path = self.args.INPUT / ("tif/" + metadata_name)
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)

            date_str = next(reader)[-1]

        # This is where the exclusion happens. If the date str contains more than one date the file is excluded here.
        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            return None, None, None

        square_km = int(split_file_name[3])
        return coords, date, square_km


if __name__ == "__main__":
    processor = SaxonyProcessor()
    processor.process()
