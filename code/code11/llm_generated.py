import requests
import bz2
import os
from eccodes import codes_grib_new_from_file, codes_get, codes_release

# Download the .bz2 file
url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/t_2m/icon-d2_germany_icosahedral_single-level_2025051000_000_2d_t_2m.grib2.bz2"
response = requests.get(url)

# Save the .bz2 file
bz2_file_path = "icon-d2_germany_icosahedral_single-level_2025051000_000_2d_t_2m.grib2.bz2"
with open(bz2_file_path, 'wb') as f:
    f.write(response.content)

# Decompress the .bz2 file
grib_file_path = bz2_file_path[:-4]  # Removing the .bz2 extension
with bz2.BZ2File(bz2_file_path, 'rb') as f_in:
    with open(grib_file_path, 'wb') as f_out:
        f_out.write(f_in.read())

# Open the GRIB file and print basic information
with open(grib_file_path, 'rb') as f:
    while True:
        try:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            
            # List available GRIB keys
            keys = codes_get(gid, 'keyList').split(',')
            print("Available GRIB keys:", keys)
            
            # Print some values
            short_name = codes_get(gid, 'shortName', missing=1)
            level = codes_get(gid, 'level', missing=1)
            print("shortName:", short_name)
            print("level:", level)

            # Release the GRIB message
            codes_release(gid)
        except Exception as e:
            print("Error:", e)
            break

# Clean up
os.remove(bz2_file_path)
os.remove(grib_file_path)