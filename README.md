''' # UAV_Classification
!pip install pyulog pandas numpy


from google.colab import files
uploaded = files.upload()

import os
ulg_files = [f for f in os.listdir('/content') if f.endswith('.ulg')]
print("ULG files found:")
print(ulg_files)
print("Total files:", len(ulg_files))

from pyulog import ULog
ulogs = []
for file in ulg_files:
    print("Loading:", file)
    ulog = ULog(file)
    ulogs.append(ulog)
print("Total logs loaded:", len(ulogs))

ulog = ulogs[0]
attitude = ulog.get_dataset('vehicle_attitude').data
print(attitude.keys())

for topic in ulogs[0].data_list:
    print(topic.name)


import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

all_data = []

mode_map = {0:'MANUAL', 3:'HOVER', 4:'CRUISE', 6:'RETURN', 7:'LAND'}

for ulog in ulogs:
    attitude = ulog.get_dataset('vehicle_attitude').data
    position = ulog.get_dataset('vehicle_local_position').data
    angular = ulog.get_dataset('vehicle_angular_velocity').data
    actuator = ulog.get_dataset('actuator_controls_0').data
    status = ulog.get_dataset('vehicle_status').data
    q0 = attitude['q[0]']
    q1 = attitude['q[1]']
    q2 = attitude['q[2]']
    q3 = attitude['q[3]']
    quat = np.column_stack((q1, q2, q3, q0))
    r = R.from_quat(quat)
    euler = r.as_euler('xyz', degrees=False)
    roll = euler[:,0]
    pitch = euler[:,1]
    yaw = euler[:,2]
    # Align all arrays to same length
    min_length = min(
        len(roll),
        len(position['vx']),
        len(angular['xyz[0]']),
        len(actuator['control[0]']),
        len(status['nav_state'])
    )
    df = pd.DataFrame({
        'roll': roll[:min_length],
        'pitch': pitch[:min_length],
        'yaw': yaw[:min_length],
        'vx': position['vx'][:min_length],
        'vy': position['vy'][:min_length],
        'vz': position['vz'][:min_length],
        'altitude': position['z'][:min_length],
        'p': angular['xyz[0]'][:min_length],
        'q': angular['xyz[1]'][:min_length],
        'r': angular['xyz[2]'][:min_length],
        'throttle': actuator['control[0]'][:min_length],
        'flight_mode': [mode_map.get(x,'OTHER') for x in status['nav_state'][:min_length]]
    })
    all_data.append(df)
# Combine all logs
final_df = pd.concat(all_data, ignore_index=True)
print("Final dataset shape:", final_df.shape)
final_df.head()

# Count how many rows we have for each flight mode
final_df['flight_mode'].value_counts()

# Remove rows with OTHER flight mode
final_df = final_df[final_df['flight_mode'] != 'OTHER']

# Check how many rows per mode
print(final_df['flight_mode'].value_counts())

# Quick look at the cleaned dataset
final_df.head()

# Save final_df to CSV
csv_filename = "uav_telemetry_dataset.csv"
final_df.to_csv(csv_filename, index=False)

print(f"Dataset saved to {csv_filename}")

from google.colab import files
files.download(csv_filename) '''

