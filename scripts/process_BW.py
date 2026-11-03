import datetime
from scripts.processing_utils import DOPProcessor


class BadenWuerttembergProcessor(DOPProcessor):

    def extract_from_file_name(self, file_name: str) -> tuple[tuple[float, float], datetime.date, int]:
        """
        An example file name would be dop20_32_534_5364_2025-08-11.tif
        From this we want to extract:
            - the coords 534, 5364 (they are divided by 1000)
            - the date 2025-08-11
            - the size of the tile which can only be 1x1.
        """
        split_file_name = file_name.split("_")
        coords = list(map(float, split_file_name[2:4]))
        date_str = split_file_name[-1].split(".")[0]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        square_km = 1

        return coords, date, square_km


if __name__ == "__main__":
    processor = BadenWuerttembergProcessor()
    processor.process()
