import datetime
import pathlib
from scripts.processing_utils import DOPProcessor


class SaxonyAnhaltProcessor(DOPProcessor):

    @staticmethod
    def parse_metadata(path: pathlib.Path):

        with open(path, "rb") as f:
            content = f.read().decode("ascii", errors="replace")
        date_str_start = content.find("Aktualitaet_Kachelinformation:")
        date_str_end = content.find("\n", date_str_start)
        line = content[date_str_start:date_str_end]
        date_str = line.split(":")[1].strip()

        return date_str

    def extract_from_file_name(self, file_name: str) -> tuple[tuple[float, float], datetime.date, int]:
        """
        An example file name would be dop20rgbi_32_642_5782_2_st_2023.tif
        From this we want to extract:
            - the coords 642, 5782 (they are divided by 1000)
            - the date is not included in the filename and will be extracted from an auxillary meta file.
            - the size of the tile which can only be 2x2 km.
        """
        split_file_name = file_name.split("_")
        coords = list(map(float, split_file_name[2:4]))
        metadata_name = file_name.replace("tif", "meta")
        metadata_path = self.args.INPUT / ("metadata/" + metadata_name)
        date_str = self.parse_metadata(metadata_path)
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        square_km = int(split_file_name[4])

        return coords, date, square_km


if __name__ == "__main__":
    processor = SaxonyAnhaltProcessor()

    processor.process()
