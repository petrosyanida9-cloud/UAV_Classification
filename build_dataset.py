import os
import numpy as np
import pandas as pd
from pyulog import ULog


# ===============================
# Quaternion → Euler conversion
# ===============================
def quaternion_to_euler(w, x, y, z):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


# ===============================
# Safe dataset getter
# ===============================
def safe_get_dataset(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d.data
    return None


# ===============================
# Extract features from single ULog
# ===============================
def extract_from_ulg(file_path, mode_label):

    print(f"Processing: {file_path}")

    try:
        ulog = ULog(file_path)
    except Exception as e:
        print(f"❌ Failed to open {file_path}: {e}")
        return None

    # Local position
    pos_data = safe_get_dataset(ulog, "vehicle_local_position")
    if pos_data is None:
        print("⚠ vehicle_local_position not found")
        return None

    # Attitude (quaternion)
    att_data = safe_get_dataset(ulog, "vehicle_attitude")
    if att_data is None:
        print("⚠ vehicle_attitude not found")
        return None

    # Angular velocity (try multiple possibilities)
    ang_data = safe_get_dataset(ulog, "vehicle_angular_velocity")

    if ang_data is None:
        ang_data = safe_get_dataset(ulog, "vehicle_rates_setpoint")

    if ang_data is None:
        print("⚠ No angular velocity topic found")
        return None

    # Acceleration
    acc_data = safe_get_dataset(ulog, "vehicle_acceleration")
    if acc_data is None:
        print("⚠ vehicle_acceleration not found")
        return None

    # ===============================
    # Synchronize by minimum length
    # ===============================
    min_len = min(
        len(pos_data["x"]),
        len(att_data["q[0]"]),
        len(ang_data[list(ang_data.keys())[1]]),
        len(acc_data[list(acc_data.keys())[1]])
    )

    # ===============================
    # Build dataframe
    # ===============================
    df = pd.DataFrame()

    df["vx"] = pos_data["vx"][:min_len]
    df["vy"] = pos_data["vy"][:min_len]
    df["vz"] = pos_data["vz"][:min_len]
    df["z"] = pos_data["z"][:min_len]

    df["ax"] = acc_data["xyz[0]"][:min_len]
    df["ay"] = acc_data["xyz[1]"][:min_len]
    df["az"] = acc_data["xyz[2]"][:min_len]

    df["p"] = ang_data[list(ang_data.keys())[1]][:min_len]
    df["q"] = ang_data[list(ang_data.keys())[2]][:min_len]
    df["r"] = ang_data[list(ang_data.keys())[3]][:min_len]

    # Quaternion → Euler
    w = att_data["q[0]"][:min_len]
    x = att_data["q[1]"][:min_len]
    y = att_data["q[2]"][:min_len]
    z_q = att_data["q[3]"][:min_len]

    roll, pitch, yaw = quaternion_to_euler(w, x, y, z_q)

    df["roll"] = roll
    df["pitch"] = pitch
    df["yaw"] = yaw

    df["flight_mode"] = mode_label

    return df


# ===============================
# Process entire folder
# ===============================
def process_folder(folder_path, mode_label):

    all_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".ulg"):
            full_path = os.path.join(folder_path, file)
            df = extract_from_ulg(full_path, mode_label)
            if df is not None:
                all_data.append(df)

    if len(all_data) == 0:
        return None

    return pd.concat(all_data, ignore_index=True)


# ===============================
# MAIN
# ===============================
mode_folders = {
    "Manual": "/home/aram/Manual_csv",
    "Stabilized": "/home/aram/Stabilized_csv",
    "Position": "/home/aram/Position_csv",
    "Altitude": "/home/aram/Altitude_csv",
    "Offboard": "/home/aram/Offboard_csv"
}

for mode, folder in mode_folders.items():

    print(f"\n=== Building {mode} dataset ===")

    df_mode = process_folder(folder, mode)

    if df_mode is not None:
        output_name = f"{mode}_dataset.csv"
        df_mode.to_csv(output_name, index=False)
        print(f"✅ Saved {output_name}")
    else:
        print(f"⚠ No valid data for {mode}")