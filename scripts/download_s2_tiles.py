from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from dotenv import load_dotenv
import pathlib
import os
import argparse
import datetime
import json
import textwrap
from tqdm import tqdm
import requests
import time

load_dotenv()
CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")
client = BackendApplicationClient(client_id=CLIENT_ID)
oauth = OAuth2Session(client=client)
collection_id = "sentinel-2-l1c"

# get an authentication token
token = oauth.fetch_token(
    token_url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    client_secret=CLIENT_SECRET,
    include_client_id=True,
)

headers_request = {"Authorization": f"Bearer {token['access_token']}"}
parser = argparse.ArgumentParser()

parser.add_argument("INPUT", type=pathlib.Path)
parser.add_argument("OUTPUT", type=pathlib.Path)
parser.add_argument(
    "-c", "--cloud", type=float, default=1, help="Fraction of cloud coverage that is considered acceptable."
)

args = parser.parse_args()

os.makedirs(args.OUTPUT, exist_ok=True)

transformer_32632 = Transformer.from_crs("EPSG:25832", "EPSG:32632", always_xy=True)
transformer_4326 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)


def request_with_retry(method, url, max_retries=5, **kwargs):
    """
    Wraps oauth.post/get with 429 handling that respects Retry-After.
    """
    global headers_request

    for attempt in range(max_retries):
        response = method(url, **kwargs)

        if response.status_code == 401:
            token = oauth.fetch_token(
                token_url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                client_secret=CLIENT_SECRET,
                include_client_id=True,
            )
            headers_request = {"Authorization": f"Bearer {token['access_token']}"}

        if response.status_code != 429:
            return response

        retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
        wait_miliseconds = float(retry_after) if retry_after else (2**attempt)

        time.sleep(wait_miliseconds / 1000)

    # last attempt — raise for status so caller's existing error handling still works
    response.raise_for_status()
    return response


def to_dateformat(date: datetime.datetime):
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_possible_dates_from_catalog(bbox_4326, from_date, to_date, date, limit=100):
    data = {
        "bbox": bbox_4326,
        "datetime": f"{to_dateformat(from_date)}/{to_dateformat(to_date)}",
        "collections": ["sentinel-2-l2a"],
        "limit": limit,
    }

    url = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
    response = request_with_retry(oauth.post, url, headers=headers_request, json=data)

    features = response.json()["features"]

    def distance_to_date(catalog_entry):
        catelog_date_str = catalog_entry

        catalog_date = datetime.datetime.fromisoformat(catelog_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        return abs(catalog_date - date)

    return sorted({f["properties"]["datetime"] for f in features}, key=distance_to_date)


def get_roi_cloud_fraction(bbox_32632, date, margin_seconds=10):
    evalscript_scl = textwrap.dedent("""\
        //VERSION=3
        function setup() {
          return {
            input: ["SCL", "dataMask"],
            output: { bands: 2, sampleType: "UINT8" }
          };
        }
        function evaluatePixel(s) {
          return [s.SCL, s.dataMask];
        }
    """)

    json_request = {
        "input": {
            "bounds": {
                "bbox": bbox_32632,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32632"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": to_dateformat(date - datetime.timedelta(seconds=margin_seconds)),
                            "to": to_dateformat(date + datetime.timedelta(seconds=margin_seconds)),
                        }
                    },
                }
            ],
        },
        "output": {
            "width": 100,
            "height": 100,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript_scl,
    }
    r = request_with_retry(
        oauth.post, "https://sh.dataspace.copernicus.eu/api/v1/process", headers=headers_request, json=json_request
    )
    r.raise_for_status()

    with rasterio.open(io.BytesIO(r.content)) as src:
        arr = src.read()  # shape (2, H, W): band 0 = SCL, band 1 = dataMask

    scl, mask = arr[0], arr[1]
    valid = mask == 1
    if valid.sum() == 0:
        return None, None  # genuinely no data for this ROI/time

    cloud_classes = {0, 1, 3, 8, 9, 10}
    is_cloud = np.isin(scl, list(cloud_classes))
    return is_cloud.mean(), scl  # fraction, computed only over valid pixels


def get_best_date_for_roi(bbox_32632, bbox_4326, from_date, to_date, flight_date, threshold=1.0):
    candidates = get_possible_dates_from_catalog(bbox_4326, from_date, to_date, flight_date)
    best_date, best_frac, best_scl = None, threshold, None
    for date_str in candidates:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        frac, scl = get_roi_cloud_fraction(bbox_32632, date=date)
        if frac is None:
            continue
        if frac <= best_frac:
            best_date, best_frac, best_scl = date, frac, scl
        if best_frac <= 1e-6:
            break

    return best_date, best_frac, best_scl


def get_process_api_request(bbox, from_date, to_date, collection_id="sentinel-2-l1c"):

    evalscript = """
    //VERSION=3
    function setup() {
    return {
        input: ["B01","B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
        output: {
        bands: 13,
        sampleType: "FLOAT32"
        }
    }
    }

    function evaluatePixel(sample) {
        return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12]
    }
    """
    json_request = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32632"},
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {"from": f"{to_dateformat(from_date)}", "to": f"{to_dateformat(to_date)}"},
                    },
                    "type": collection_id,
                    "processing": {"harmonizeValues": "true"},
                }
            ],
        },
        "output": {
            "width": 100,
            "height": 100,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    return json_request


def get_tile(bbox_32632, date):
    from_date = date - datetime.timedelta(seconds=5)
    to_date = date + datetime.timedelta(seconds=5)

    json_request = get_process_api_request(bbox_32632, from_date, to_date, collection_id)

    r = request_with_retry(
        oauth.post, "https://sh.dataspace.copernicus.eu/api/v1/process", headers=headers_request, json=json_request
    )
    try:
        r.raise_for_status()
    except Exception as e:
        print(e)
        raise e

    return r.content


for file in tqdm(sorted(args.INPUT.rglob("*.tif"))):
    file_name = file.name
    output_path = args.OUTPUT / file_name
    log_file_path = args.OUTPUT / (datetime.datetime.now().isoformat() + ".txt")

    if output_path.exists():
        continue

    margin_days = 7
    split_file_name = file_name.split("_")
    coords = list(map(float, split_file_name[2:4]))
    date = split_file_name[-1].split(".")[0]
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    from_date = date - datetime.timedelta(days=margin_days)
    to_date = date + datetime.timedelta(days=margin_days)

    # Min easting / northing
    min_e = coords[0] * 1000
    min_n = coords[1] * 1000
    # Tiles are 1x1 km^2
    max_e = min_e + 1000
    max_n = min_n + 1000
    min_lon, min_lat = transformer_32632.transform(min_e, min_n)
    max_lon, max_lat = transformer_32632.transform(max_e, max_n)
    bbox_32632 = [min_lon, min_lat, max_lon, max_lat]

    min_lon, min_lat = transformer_4326.transform(min_e, min_n)
    max_lon, max_lat = transformer_4326.transform(max_e, max_n)
    bbox_4326 = [min_lon, min_lat, max_lon, max_lat]

    best_date, best_frac, best_scl = get_best_date_for_roi(
        bbox_32632=bbox_32632, bbox_4326=bbox_4326, from_date=from_date, to_date=to_date, flight_date=date
    )
    if best_scl is None:
        # No tile was found matching the cloud coverage
        with open(log_file_path, "w") as f:
            f.write(f"{file_name},{best_frac}, {args.cloud}\n")

    best_scl = best_scl.astype(np.uint8)[np.newaxis, :, :]
    dst_crs = "EPSG:25832"
    response = get_tile(bbox_32632, best_date)

    with rasterio.open(io.BytesIO(response)) as tif_tile:
        tile = tif_tile.read()

        # REFLECTANCE_SAT = 0.3  # reflectance value that maps to 255 ("burned out" white)

        # refl_uint8 = np.clip(tile / REFLECTANCE_SAT, 0, 1)
        # refl_uint8 = (refl_uint8 * 255).astype(np.uint8)

        transform, width, height = calculate_default_transform(
            tif_tile.crs, dst_crs, tif_tile.width, tif_tile.height, *tif_tile.bounds
        )

        profile = tif_tile.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": rasterio.float32,
                "count": profile["count"] + 1,
            }
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        # dst.write(np.concatenate([refl_uint8, best_scl], axis=0))
        dst.write(np.concatenate([tile, best_scl], axis=0))
        dst.update_tags(cloud_cover_fraction=best_frac)
        dst.set_band_description(dst.count, "SCL")
        dst.update_tags(recording_date=best_date.isoformat())
