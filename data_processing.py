# Processing Data
# Last updated 2nd May 2024

import os
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np

### set working directory
os.chdir("C:/Users/angel/OneDrive/Documents/GitHub/RVF-ABM-April_2024")

# Prevalence data
prev_data = pd.read_excel("data/naddec_04022024.xls")
### Make district lower case
prev_data['district'] = prev_data['district'].str.lower()
### filter animal to be "Bovine"
prev_data = prev_data[prev_data['host'] == 'Bovine']
### filter date_sample  within 2021
prev_data['date_sample'] = pd.to_datetime(prev_data['date_sample'], format='mixed')
prev_data = prev_data[prev_data['date_sample'].dt.year == 2021]
## Find prevalence
prev_data = prev_data[(prev_data['district'].isin(['kampala', 'kiruhura']))]


# Population data
pop_data = pd.read_excel("data/Cattle_2021.xlsx")
### Make district lower case
pop_data['district'] = pop_data['district'].str.lower()
### Remove extra white spaces
pop_data['district'] = pop_data['district'].str.strip().str.replace(r'\s+', ' ')


## Ratio of cattle in kiruhura vs kampala
kampala_number = pop_data.loc[pop_data['district'] == 'kampala', 'number'].iloc[0]
kiruhura_number = pop_data.loc[pop_data['district'] == 'kiruhura', 'number'].iloc[0]
## Calculate the probability of the cow being in kampala
p_kampala = kampala_number / (kampala_number + kiruhura_number)

# Movement data
movement_data = pd.read_excel("data/livestock_all.xlsx")
### Make district lower case and remove extra white spaces
movement_data['origin'] = movement_data['origin'].str.lower()
movement_data['origin'] = movement_data['origin'].str.strip().str.replace(r'\s+', ' ')
movement_data['destination'] = movement_data['destination'].str.lower()
movement_data['destination'] = movement_data['destination'].str.strip().str.replace(r'\s+', ' ')
### Filter for bovine
movement_data = movement_data[movement_data['species'] == 'BOVINE']
### Filter date_issue for 2021: Won't do because there is no movemen btwn Kampala and kiruhura in 2021
movement_data['date_issue'] = pd.to_datetime(movement_data['date_issue'], format='mixed')
movement_data = movement_data[movement_data['date_issue'].dt.year == 2021]
## Get probability of movement
### First find the total number of cattle that were moved.
quantity_kampala_kiruhura = movement_data.loc[(movement_data['origin'] == 'kampala') & (movement_data['destination'] == 'kiruhura'), 'quantity'].sum()
quantity_kiruhura_kampala = movement_data.loc[(movement_data['origin'] == 'kiruhura') & (movement_data['destination'] == 'kampala'), 'quantity'].sum()
### Find probability by dividing total
prob_kampala_kiruhura = quantity_kampala_kiruhura / kampala_number
prob_kiruhura_kampala = quantity_kiruhura_kampala / kiruhura_number

# Temperature data
# Load district boundaries
district_boundaries = gpd.read_file("data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm3_ubos_20200824.shp")

