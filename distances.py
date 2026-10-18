import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit


# 1. Load your raw datasets
housing = pd.read_csv("final_house.csv")
subway = pd.read_csv("mta_s.csv")

def latlon_to_cartesian(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    R = 3958.8  # Earth's radius in miles
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.column_stack((x, y, z))

housing_3d = latlon_to_cartesian(housing['Latitude'], housing['Longitude'])
subway_3d = latlon_to_cartesian(subway['GTFS Latitude'], subway['GTFS Longitude'])

# 3. Build spatial KD-Tree for ultra-fast nearest-neighbor search
tree = KDTree(subway_3d)

# 4. Query nearest subway station for each housing project
distances_miles, nearest_indices = tree.query(housing_3d, k=1)

# 5. Attach results to DataFrame
housing['subway_dist_miles'] = np.round(distances_miles, 2)
housing['nearest_station_id'] = nearest_indices

# 6. Save enriched CSV for R Quarto
# housing.to_csv("housing_with_distances.csv", index=False)
# print("Successfully computed nearest subway distances!")

# 1. Clean data (handle log(0) / negative / missing values)
valid_mask = (housing['Percent_Low_Income'] > 0) & housing['subway_dist_miles'].notna()
clean_housing = housing[valid_mask]

distance = clean_housing['subway_dist_miles'].values
low_inc = np.log(clean_housing['Percent_Low_Income'].values)

# 2. Define non-linear model function
def our_nonlinear_model(x, a, b):
    """
    x : the x-values of our data (low_inc)
    a, b : the parameters of our model
    """
    return a * np.exp(b * x)

# 3. Fit curve (x = low_inc, y = distance)
popt, pcov = curve_fit(our_nonlinear_model, low_inc, distance, p0=[1.0, 1.0])
a, b = popt

# Sort x-values for a smooth model line on the plot
sort_idx = np.argsort(low_inc)
low_inc_sorted = low_inc[sort_idx]
non_linear_fit_sorted = our_nonlinear_model(low_inc_sorted, *popt)

# Fitted values matching point-for-point for residual calculation
non_linear_fit = our_nonlinear_model(low_inc, *popt)
residuals = distance - non_linear_fit

# 4. Multi-panel subplot with residuals
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 7), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)

# Top Panel: Data and Fitted Line
ax[0].scatter(low_inc, distance, alpha=0.6, label='Observed Data')
ax[0].plot(low_inc_sorted, non_linear_fit_sorted, 'k-', label='Best-fit Curve')
ax[0].set_ylabel('Distance from Subway (Miles)', fontsize=14)
ax[0].tick_params(axis='both', labelsize=12)
ax[0].legend()

# Bottom Panel: Residuals
ax[1].axhline(y=0, color='gray', linestyle='solid')
ax[1].scatter(low_inc, residuals, alpha=0.6)
ax[1].set_xlabel('Log(Percent Low Income Units)', fontsize=14)
ax[1].set_ylabel('Residuals', fontsize=12)
ax[1].tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()