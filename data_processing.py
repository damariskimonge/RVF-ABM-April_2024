# Processing Data
# Last updated 2nd May 2024

import os
import geopandas as gpd
#import rasterio
#import rasterio.mask  
#from rasterio.windows import from_bounds
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
## Sort by district name: alphabetical order
pop_data = pop_data.sort_values(by='district').reset_index(drop=True)
## Total population
total_pop = pop_data['number'].sum()
# Calculate the probability of each cow being assigned to each district
pop_data['probability'] = pop_data['number'] / total_pop
# Create the probability array
districts = pop_data['district'].values
district_probabilities = pop_data['probability'].values


# Movement data
movement_data = pd.read_excel("data/livestock_all.xlsx")
### Make district lower case and remove extra white spaces
movement_data['origin'] = movement_data['origin'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
movement_data['destination'] = movement_data['destination'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
### Filter for bovine
movement_data = movement_data[movement_data['species'] == 'BOVINE']
### Filter date_issue for 2021.
movement_data.loc[:, 'date_issue'] = pd.to_datetime(movement_data['date_issue'], format='mixed')
movement_data = movement_data[movement_data['date_issue'].dt.year == 2021]
## Get probability of movement: create a probability matrix
### Get unique districts
unique_districts = sorted(pop_data['district'].unique())
### Initialize the probability matrix
movement_prob_matrix = pd.DataFrame(0, index=unique_districts, columns=unique_districts, dtype=float)
### Calculate the total movements from each district
total_movements = movement_data.groupby('origin')['quantity'].sum()
### Calculate movement probabilities
for _, row in movement_data.iterrows():
    origin = row['origin']
    destination = row['destination']
    quantity = row['quantity']
    if origin in unique_districts and destination in unique_districts:
        movement_prob_matrix.loc[origin, destination] += quantity
### Normalize the probabilities based on the total population in the origin district
for origin in unique_districts:
    if total_movements.get(origin, 0) > 0:
        district_population = pop_data[pop_data['district'] == origin]['number'].values[0]
        movement_prob_matrix.loc[origin] /= district_population
### Replace NaNs with 0
movement_prob_matrix.fillna(0, inplace=True)
# Calculate the probability of staying in the same district
for origin in unique_districts:
    total_prob = movement_prob_matrix.loc[origin].sum()
    if total_prob < 1:
        stay_probability = 1 - total_prob
        movement_prob_matrix.loc[origin, origin] += stay_probability
# Handle districts with no movement at all
for origin in unique_districts:
    if total_movements.get(origin, 0) == 0:
        movement_prob_matrix.loc[origin, origin] = 1
# Ensure all probabilities sum to 1
for origin in unique_districts:
    row_sum = movement_prob_matrix.loc[origin].sum()
    if row_sum > 0:
        movement_prob_matrix.loc[origin] /= row_sum
# Replace NaNs with 0 (if any)
movement_prob_matrix.fillna(0, inplace=True)
# Convert to numpy array for use in the simulation
movement_prob_array = movement_prob_matrix.to_numpy()



# Load the district data 
district_data = gpd.read_file('data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm2_ubos_20200824.shp')
### Make the district names lower case
district_data['ADM2_EN'] = district_data['ADM2_EN'].str.lower()

# Load the environment data
env_data = pd.read_excel("data/env_data.xlsx")
env_data['district'] = env_data['district'].str.lower().str.strip()

# Create dictionaries to store the environmental data arrays
env_arrays = {}
for district in env_data['district'].unique():
    veg = env_data.loc[env_data['district'] == district, 'vegetation_index'].iloc[0]
    temp = env_data.loc[env_data['district'] == district, 'temperature'].iloc[0]
    rain = env_data.loc[env_data['district'] == district, 'precipitation'].iloc[0]
    
    env_arrays[district] = {
        'veg_arr': np.random.normal(loc=veg, size=365),
        'temp_arr': np.random.normal(loc=temp, size=365),
        'rain_arr': np.random.normal(loc=rain, size=365),
    }
