import re
import os
import bz2
import requests
import wget

base_url = "https://opendata.dwd.de/weather/nwp/icon/grib/00/"
clat_path = "clat/"
clon_path = "clon/"

# Function to find the latest available timestamp from DWD server
def get_latest_timestamp(path):
    listing_url = base_url + path
    response = requests.get(listing_url)
    if response.status_code != 200:
        raise RuntimeError(f"Could not fetch listing: {listing_url}")
    timestamps = re.findall(
        r'icon_global_icosahedral_time-invariant_(\d{10})_CLAT\.grib2\.bz2',
        response.text
    )
    return max(timestamps) if timestamps else None

# Get the latest available timestamp
timestamp = get_latest_timestamp(clat_path)
if not timestamp:
    raise RuntimeError("Could not determine latest timestamp from DWD server.")

files = {
    "clat": f"clat/icon_global_icosahedral_time-invariant_{timestamp}_CLAT.grib2.bz2",
    "clon": f"clon/icon_global_icosahedral_time-invariant_{timestamp}_CLON.grib2.bz2"
}

rename_map = {"clat": "icon_lat.grib", "clon": "icon_lon.grib"}

for key, path in files.items():
    filename = os.path.basename(path)
    url = base_url + path

    print(f"Downloading {url} ...")
    wget.download(url, filename)
    print(f"\nDownloaded {filename}")

    # Uncompress the .bz2 file
    with bz2.open(filename, 'rb') as compressed, open(filename[:-4], 'wb') as out_file:
        out_file.write(compressed.read())
    print(f"Decompressed {filename} to {filename[:-4]}")

    # Rename the extracted file
    extracted_filename = filename[:-4]  # Remove .bz2
    new_filename = rename_map[key]
    os.rename(extracted_filename, new_filename)
    print(f"Renamed {extracted_filename} to {new_filename}")
