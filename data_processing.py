# Processing Data
# Last updated 2nd May 2024

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np

# Prevalence data


# Population data


# Movement data


# Temperature data
# Load district boundaries
districts = gpd.read_file("C:/Users/angel/OneDrive/Documents/GitHub/RVF-ABM-April_2024/data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm3_ubos_20200824.shp")
# Load empty df
jan_df = pd.DataFrame(columns=['district', 'temperature'])
# Open temperature TIF file
with rasterio.open("C:/Users/angel/OneDrive/Documents/GitHub/RVF-ABM-April_2024/data/temp/wc2.1_10m_prec_2021-01.tif") as src:
    # Loop through each district
    for index, district in districts.iterrows():
        # Extract geometry of the district
        district_geometry = district['geometry']
        
        # Get bounding box of the district
        minx, miny, maxx, maxy = district_geometry.bounds
        
        # Create a window based on the bounding box
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        # Read temperature data for the district's window
        district_data = src.read(1, masked=True, window=window)
        
        # Average the temperature       
        average_temperature = district_data.mean()
        rounded_temperature = np.round(average_temperature)  # Round to 2 decimal places
        
        # Store results in the DataFrame
        new_row = pd.DataFrame({'district': [district['ADM3_EN']], 'temperature': [rounded_temperature]})
        jan_df = pd.concat([jan_df, new_row], ignore_index=True)


        # will need to become a for loop for each temperature file