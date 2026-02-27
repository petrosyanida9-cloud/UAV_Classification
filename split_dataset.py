# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# # Load balanced dataset
# df = pd.read_csv("/home/aram/Downloads/balanced_dataset.csv")
#
# # Features and label
# FEATURES = [
#     'vx', 'vy', 'vz', 'z',
#     'ax', 'ay', 'az',
#     'p', 'q', 'r',
#     'roll', 'pitch', 'yaw'
# ]
# LABEL = 'flight_mode'
#
# # Assign fake 'log_id' if you don't have original log info:
# # Here we split each flight_mode into 5 "logs" for demonstration
# logs_per_mode = 5
# rows_per_log = df.groupby(LABEL).size().iloc[0] // logs_per_mode
#
# log_id_list = []
# for mode in df[LABEL].unique():
#     start_idx = 0
#     for i in range(logs_per_mode):
#         end_idx = start_idx + rows_per_log
#         log_id_list.extend([f"{mode}_log{i+1}"] * rows_per_log)
#         start_idx = end_idx
#     # Add remaining rows to last log
#     remainder = df[df[LABEL]==mode].shape[0] - len([x for x in log_id_list if x.startswith(mode+"_log")])
#     log_id_list.extend([f"{mode}_log{logs_per_mode}"] * remainder)
#
# df['log_id'] = log_id_list
#
# # Now split by log
# train_logs = []
# val_logs = []
# test_logs = []
#
# np.random.seed(42)
# for mode in df[LABEL].unique():
#     logs = df[df[LABEL]==mode]['log_id'].unique()
#     np.random.shuffle(logs)
#     n = len(logs)
#     train_logs += logs[:int(0.6*n)].tolist()
#     val_logs   += logs[int(0.6*n):int(0.8*n)].tolist()
#     test_logs  += logs[int(0.8*n):].tolist()
#
# train_df = df[df['log_id'].isin(train_logs)].copy()
# val_df   = df[df['log_id'].isin(val_logs)].copy()
# test_df  = df[df['log_id'].isin(test_logs)].copy()
#
# # Remove log_id column for ML
# train_df.drop(columns=['log_id'], inplace=True)
# val_df.drop(columns=['log_id'], inplace=True)
# test_df.drop(columns=['log_id'], inplace=True)
#
# # Normalize features (fit scaler on train only)
# scaler = StandardScaler()
# train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
# val_df[FEATURES]   = scaler.transform(val_df[FEATURES])
# test_df[FEATURES]  = scaler.transform(test_df[FEATURES])
#
# # Save final splits
# train_df.to_csv("/home/aram/Downloads/train_dataset.csv", index=False)
# val_df.to_csv("/home/aram/Downloads/val_dataset.csv", index=False)
# test_df.to_csv("/home/aram/Downloads/test_dataset.csv", index=False)
#
# print("Train/Val/Test shapes:", train_df.shape, val_df.shape, test_df.shape)
# print("Flight mode distribution (train):\n", train_df[LABEL].value_counts())

#this is single mode Manual file
#/home/aram/Manual_csv/d3b324da-5504-48ff-b613-e05be010063e.ulg

#this is single mode Stabilized file
#/home/aram/Stabilized_csv/6cb1b7d6-6ec3-411f-b48d-60bc2b088ce9.ulg

#this is a file with Manual, Altitude, Position flight modes in it , mixture of my 3 labels
#/home/aram/Downloads/4624b22c-2e0d-42f3-ae82-7f56645cbbf0.ulg

#and finally this is a file that has Manual, Altitude, Position, Mission, Return to Land, Land flight modes in it , so a mixture + Other label modes
#/home/aram/Downloads/1c8fbb58-af8b-403f-9178-e12917494a8a.ulg

# /home/aram/Downloads/val_dataset.csv
# /home/aram/Downloads/train_dataset.csv
# /home/aram/Downloads/test_dataset.csv
# /home/aram/Downloads/balanced_dataset.csv
# /home/aram/Downloads/full_dataset.csv
# /home/aram/Downloads/Offboard_dataset.csv
# /home/aram/Downloads/Altitude_dataset.csv
# /home/aram/Downloads/Position_dataset.csv
# /home/aram/Downloads/Stabilized_dataset.csv
# /home/aram/Downloads/Manual_dataset.csv

##########################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ===============================
# 1. LOAD ORIGINAL DATASET
# ===============================

df = pd.read_csv("/home/aram/PyCharmMiscProject/merged_dataset_labeled1.csv")

FEATURES = [
    'vx', 'vy', 'vz', 'z',
    'ax', 'ay', 'az',
    'p', 'q', 'r',
    'roll', 'pitch', 'yaw'
]

LABEL = "flight_mode"
FLIGHT_ID = "id"

# ===============================
# 2. FLIGHT-LEVEL SPLIT
# ===============================

unique_flights = df[[FLIGHT_ID, LABEL]].drop_duplicates()

train_ids, temp_ids = train_test_split(
    unique_flights,
    test_size=0.3,
    random_state=42,
    stratify=unique_flights[LABEL]
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.5,
    random_state=42,
    stratify=temp_ids[LABEL]
)

train_df = df[df[FLIGHT_ID].isin(train_ids[FLIGHT_ID])].copy()
val_df   = df[df[FLIGHT_ID].isin(val_ids[FLIGHT_ID])].copy()
test_df  = df[df[FLIGHT_ID].isin(test_ids[FLIGHT_ID])].copy()

print("Flights per split:")
print("Train:", train_df[FLIGHT_ID].nunique())
print("Val:", val_df[FLIGHT_ID].nunique())
print("Test:", test_df[FLIGHT_ID].nunique())

# ===============================
# 3. FEATURE SCALING (TRAIN ONLY)
# ===============================

scaler = StandardScaler()

train_df.loc[:, FEATURES] = scaler.fit_transform(train_df[FEATURES])
val_df.loc[:, FEATURES]   = scaler.transform(val_df[FEATURES])
test_df.loc[:, FEATURES]  = scaler.transform(test_df[FEATURES])

joblib.dump(scaler, "/home/aram/Downloads/uav_scaler_v2.pkl")

# ===============================
# 4. SAVE FINAL SPLITS
# ===============================

train_df.to_csv("/home/aram/Downloads/train_dataset_v2.csv", index=False)
val_df.to_csv("/home/aram/Downloads/val_dataset_v2.csv", index=False)
test_df.to_csv("/home/aram/Downloads/test_dataset_v2.csv", index=False)

print("\nRow distribution (TRAIN):")
print(train_df[LABEL].value_counts())

print("\nRow distribution (VAL):")
print(val_df[LABEL].value_counts())

print("\nRow distribution (TEST):")
print(test_df[LABEL].value_counts())

print("\nDONE — clean flight-level split completed.")

# /home/aram/Downloads/test_dataset_v2.csv
# /home/aram/Downloads/val_dataset_v2.csv
# /home/aram/Downloads/train_dataset_v2.csv