# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:39:27 2024
For V distribution
@author: Nasser
"""
#%% Imports and initial settings

# for compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function
import trackpy as tp
from scipy.optimize import curve_fit
from scipy.stats import t, linregress

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
# Enable %matplotlib qt for pop-out plots in Spyder
%matplotlib qt
# Update matplotlib parameters for consistent font usage
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Latin Modern Roman",
    "font.sans-serif": ["Helvetica"]
})



plt.rc('figure', figsize=(8, 8))
plt.rc('image', cmap='gray')


# Define path
str_path = (r' ')
path = Path(str_path)

def read_experiment_parameters(folder):
    params_path = Path(folder) / 'fitted_parameters.csv'
    params_df = pd.read_csv(params_path)
    dt = float(params_df[params_df['Parameter'] == 'dt (s)']['Value'])
    mag = float(params_df[params_df['Parameter'] == 'resolution (um/pix)']['Value'])
    return dt, mag

params = read_experiment_parameters(str_path)
dt = params[0]
mag= params[1]

trajectories = pd.read_csv(path / 'trajectories.csv')

# Add these lines at the beginning of script to create the 'final' and 'segments' subfolders
path1 = path / 'velocity'
path1.mkdir(parents=True, exist_ok=True)

#%% Compute velocities

# Function to compute velocities
def compute_velocities(trajectories, dt, mag):
    velocities = []
    for particle, group in trajectories.groupby('particle'):
        group = group.sort_values(by='frame')
        vx = np.diff(group['x']) / dt * mag * 60
        vy = np.diff(group['y']) / dt * mag * 60
        v = np.sqrt(vx**2 + vy**2)
        for i in range(len(vx)):
            velocities.append({'particle': particle, 'vx': vx[i], 'vy': vy[i], 'v': v[i]})
    return pd.DataFrame(velocities)

velocities_df = compute_velocities(trajectories, dt, mag)

# Save velocities to Excel
velocities_path = path1 / 'velocities.xlsx'
velocities_df.to_excel(velocities_path, index=False)

#%% Compute mean speed for each particle

mean_speeds = velocities_df.groupby('particle')['v'].mean().reset_index()
mean_speeds.columns = ['particle', 'mean_speed']

# Save mean speeds to Excel
mean_speeds_path = path1 / 'mean_speeds.xlsx'
mean_speeds.to_excel(mean_speeds_path, index=False)


#%% Plot velocity distribution

plt.figure(figsize=(6, 5))
plt.hist(mean_speeds['mean_speed'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel(r'${\mathrm{V}}$ [µm/min]', fontsize=16)
plt.ylabel('Count', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(left=0)
plt.ylim(bottom=0)


plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = path1 / 'velocity_distribution.png'
plt.savefig(plot_path, dpi=200)

# Show the plot
plt.show()


#%% go through all folders

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

"""
define a base directory to iterate through all folders
"""
# Base directory to search
base_dir = r' '

# Directory to save the results
save_dir = r' '
os.makedirs(save_dir, exist_ok=True)

# Initialize list to store all mean speeds
all_mean_speeds = []

# Traverse the directory tree
for root, dirs, files in os.walk(base_dir):
    if 'final2' in dirs:
        final2_path = os.path.join(root, 'final2')
        mean_speeds_path = os.path.join(final2_path, 'mean_speeds.xlsx')
        if os.path.exists(mean_speeds_path):
            mean_speeds_df = pd.read_excel(mean_speeds_path)
            all_mean_speeds.append(mean_speeds_df)

# Concatenate all mean speeds into one DataFrame
if all_mean_speeds:
    all_mean_speeds_df = pd.concat(all_mean_speeds, ignore_index=True)
    
    # Save the concatenated DataFrame with all mean speeds to a CSV file
    all_mean_speeds_path = os.path.join(save_dir, 'all_mean_speeds.csv')
    all_mean_speeds_df.to_csv(all_mean_speeds_path, index=False)
    
    # Calculate the total mean and SEM across all mean speeds
    total_mean = all_mean_speeds_df['mean_speed'].mean()
    total_sem = all_mean_speeds_df['mean_speed'].sem()
    
    # Save the total mean and SEM
    summary_df = pd.DataFrame({
        'Total Mean': [total_mean],
        'Total SEM': [total_sem]
    })
    summary_path = os.path.join(save_dir, 'summary_speeds.xlsx')
    summary_df.to_excel(summary_path, index=False)
    
    print(f"Number of valid experiments: {len(all_mean_speeds)}")
    print(f"All mean speeds saved to {all_mean_speeds_path}")
    print(f"Total mean and SEM saved to {summary_path}")
else:
    print("No valid 'final2' folders found with 'mean_speeds.xlsx' file.")

#%% Plot the distribution of all mean speeds

plt.figure(figsize=(6, 5))
plt.hist(all_mean_speeds_df['mean_speed'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel(r'${\mathrm{V}}$ [µm/min]', fontsize=16)
plt.ylabel('Count', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(save_dir, 'all_mean_speeds_distribution.png')
plt.savefig(plot_path, dpi=200)

# Show the plot
plt.show()
