import pathlib
import csv
import datetime

path = pathlib.Path("/NAS/data/dop/SX/tif")
weird_csv_count = 0
all_csv_count = 0
for file in path.rglob("*.csv"):
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)

        date_str = next(reader)[-1]
        all_csv_count += 1

        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            weird_csv_count += 1
            print(weird_csv_count, date_str)
print(weird_csv_count / all_csv_count)
