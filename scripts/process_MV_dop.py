import datetime
from scripts.processing_utils import DOPProcessor


class MecklenburgVorpommernProcessor(DOPProcessor):

    def extract_from_file_name(self, file_name: str) -> tuple[tuple[float, float], datetime.date, int]:
        """
        An example file name would be dop20rgbi_33_400_6022_2_mv.tif
        From this we want to extract:
            - the coords 40, 6022 (they are divided by 1000)
            - the date 2026-04-08
            - the size of the tile which is always 2.
        """
        split_file_name = file_name.split("_")
        coords = list(map(float, split_file_name[2:4]))
        date_str = split_file_name[-1].split(".")[0]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        square_km = int(split_file_name[4])

        return coords, date, square_km


if __name__ == "__main__":
    processor = MecklenburgVorpommernProcessor()
    processor.process()
