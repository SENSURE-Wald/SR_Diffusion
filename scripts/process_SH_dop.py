import datetime
import json
from scripts.processing_utils import DOPProcessor


class SchleswigHolsteinProcessor(DOPProcessor):

    def __init__(self):
        super().__init__()

        json_path = self.args.INPUT / "SH.geojson"
        with open(json_path, "r") as f:
            data = json.load(f)

        features = data["features"]
        transformed_dict = {x["properties"]["kachel"]: x["properties"]["datum"] for x in features}
        self.date_dict = transformed_dict

    def extract_from_file_name(self, file_name: str) -> tuple[tuple[float, float], datetime.date, int]:
        """
        An example file name would be dop20rgbi_32_455_6079_1_sh_2024.tif
        From this we want to extract:
            - the coords 455, 6079 (they are divided by 1000)
            - the date is not included in the filename and will be extracted from an auxillary geojson file.
            - the size of the tile which can only be 1x1.
        """
        split_file_name = file_name.split("_")
        coords = list(map(float, split_file_name[2:4]))
        date_key = f"32{int(coords[0])}{int(coords[1])}"
        date_str = self.date_dict[date_key]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        square_km = int(split_file_name[4])

        return coords, date, square_km


if __name__ == "__main__":
    processor = SchleswigHolsteinProcessor()
    processor.process()
