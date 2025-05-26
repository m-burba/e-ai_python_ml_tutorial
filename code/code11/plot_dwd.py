from eccodes import codes_grib_new_from_file, codes_get_array, codes_release
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

lat_file = "clat.grib2"
lon_file = "clon.grib2"
data_file = "data.grib2"

def extract_values(filename, short_name):
    with open(filename, "rb") as f:
        while True:
            try:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    break
                if codes_get(gid, "shortName") == short_name:
                    values = codes_get_array(gid, "values")
                    codes_release(gid)
                    return values
            except Exception:
                break
    return None

tlat = extract_values(lat_file, "tlat")
tlon = extract_values(lon_file, "tlon")
temperature = extract_values(data_file, "2t")

if tlat is not None and tlon is not None and temperature is not None:
    temperature_c = temperature - 273.15
    mask = (temperature_c >= -10) & (temperature_c <= 50)
    
    tlat_masked = tlat[mask]
    tlon_masked = tlon[mask]
    temperature_masked = temperature_c[mask]

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    scatter = ax.scatter(tlon_masked, tlat_masked, c=temperature_masked, cmap='coolwarm', s=1, transform=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    plt.colorbar(scatter, ax=ax, orientation='vertical', label='Temperature (°C)')
    plt.title('2-Meter Temperature')
    plt.savefig('t2m.png', dpi=300)
    plt.show()
else:
    print("Error: Unable to extract one or more required datasets.")