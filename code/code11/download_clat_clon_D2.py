import requests
import bz2

urls = [
    "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/clat/icon-d2_germany_icosahedral_time-invariant_2025051000_000_0_clat.grib2.bz2",
    "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/clon/icon-d2_germany_icosahedral_time-invariant_2025051000_000_0_clon.grib2.bz2"
]

filenames = ['clat.grib2.bz2', 'clon.grib2.bz2']

for url, filename in zip(urls, filenames):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

for filename in filenames:
    with bz2.BZ2File(filename, 'rb') as f:
        data = f.read()
    with open(filename[:-4], 'wb') as f:
        f.write(data)