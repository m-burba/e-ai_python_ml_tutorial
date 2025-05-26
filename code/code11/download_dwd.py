import requests
import bz2
import eccodes

# Download the file
url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/t_2m/icon-d2_germany_icosahedral_single-level_2025051000_000_2d_t_2m.grib2.bz2"
response = requests.get(url)

# Save the .bz2 file
with open("data.grib2.bz2", "wb") as f:
    f.write(response.content)

# Decompress the .bz2 file
with open("data.grib2", "wb") as f:
    f.write(bz2.decompress(open("data.grib2.bz2", "rb").read()))

# List keys in the first GRIB message
with open("data.grib2", "rb") as f:
    with eccodes.Codec(f) as gid:
        gid.seek(0)  # Go to the start of the file
        while gid.read() is not None:
            keys = gid.keys()
            print(keys)
            break  # Only print for the first message