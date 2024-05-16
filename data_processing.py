# Processing Data
# Last updated 2nd May 2024

import os
import geopandas as gpd
import rasterio
import rasterio.mask  
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np

### set working directory
os.chdir("C:/Users/angel/OneDrive/Documents/cema/Github/RVF-ABM-April_2024")

## Creating a date object for plotting:
from datetime import datetime, timedelta
# Define the start and end dates for 2021
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 12, 31)
# Calculate the number of days between the start and end dates
num_days = (end_date - start_date).days
# Create a list of daily dates
date_list = [start_date + timedelta(days=x) for x in range(num_days + 1)]

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
### TO DO: Vera suggests using relative risk.


# Population data
pop_data = pd.read_excel("data/Cattle_2021.xlsx")
### Make district lower case
pop_data['district'] = pop_data['district'].str.lower()
### Remove extra white spaces
pop_data['district'] = pop_data['district'].str.strip().str.replace(r'/s+', ' ')
## Total population
total_pop = pop_data['number'].sum()
## Ratio of cattle in kiruhura vs kampala
kampala_number = pop_data.loc[pop_data['district'] == 'kampala', 'number'].iloc[0]
kiruhura_number = pop_data.loc[pop_data['district'] == 'kiruhura', 'number'].iloc[0]
### Population of kampala and kiruhura
pop_kam_kir = kampala_number + kiruhura_number
## Calculate the probability of the cow being in kampala
p_kampala = kampala_number / pop_kam_kir
p_kiruhura = kiruhura_number / pop_kam_kir


# Movement data
movement_data = pd.read_excel("data/livestock_all.xlsx")
### Make district lower case and remove extra white spaces
movement_data['origin'] = movement_data['origin'].str.lower()
movement_data['origin'] = movement_data['origin'].str.strip().str.replace(r'/s+', ' ')
movement_data['destination'] = movement_data['destination'].str.lower()
movement_data['destination'] = movement_data['destination'].str.strip().str.replace(r'/s+', ' ')
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

# Load the district data 
districts = gpd.read_file('data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm2_ubos_20200824.shp')
### Make the district names lower case
districts['ADM2_EN'] = districts['ADM2_EN'].str.lower()

# Load the environment data
env_data = pd.read_excel("data/env_data.xlsx")
env_data['district'] = env_data['district'].str.lower()

kampala_veg = env_data.loc[env_data['district'] == "kampala", 'vegetation_index'].iloc[0]
kampala_veg_arr = np.random.normal(loc=kampala_veg, size=365)
kampala_temp = env_data.loc[env_data['district'] == 'kampala', 'temperature'].iloc[0]
kampala_temp_arr = np.random.normal(loc=kampala_temp, size=365)
kampala_rain = env_data.loc[env_data['district'] == 'kampala', 'precipitation'].iloc[0]
kampala_rain_arr = np.random.normal(loc=kampala_rain, size=365)

kiruhura_veg = env_data.loc[env_data['district'] == "kiruhura", 'vegetation_index'].iloc[0]
kiruhura_veg_arr = np.random.normal(loc=kiruhura_veg, size=365)
kiruhura_temp = env_data.loc[env_data['district'] == 'kiruhura', 'temperature'].iloc[0]
kiruhura_temp_arr = np.random.normal(loc=kiruhura_temp , size=365)
kiruhura_rain = env_data.loc[env_data['district'] == 'kiruhura', 'precipitation'].iloc[0]
kiruhura_rain_arr = np.random.normal(loc=kiruhura_rain, size=365)

