import os
import glob
import numpy as np
import pandas as pd
from pyulog import ULog

# -----------------------------------
# Automatically collect all logs
# -----------------------------------

LOG_FILES = [

# ---------------- Manual ----------------
"/home/aram/Manual_csv/00bd805f-9a23-4b20-a364-ef84010ae52a.ulg",
"/home/aram/Manual_csv/9be3936f-e0da-4e35-9a71-4c939b9fd6a8.ulg",
"/home/aram/Manual_csv/9e67abc1-c3d9-41df-8acc-2bd594e97f32.ulg",
"/home/aram/Manual_csv/21b8a2cd-3b83-4a89-92f0-8663b9ff205c.ulg",
"/home/aram/Manual_csv/23e8f3dc-5c0f-4128-a776-12f9a319e0f2.ulg",
"/home/aram/Manual_csv/8990739c-c5dd-46fe-9d88-ba27972b1a0c.ulg",
"/home/aram/Manual_csv/a2c1c485-8c50-4d7d-9358-9f5e67d92246.ulg",
"/home/aram/Manual_csv/cab20207-6248-48ea-978d-644d630608a6.ulg",
"/home/aram/Manual_csv/d20991e6-f906-4f01-9c26-363a980ab6f6.ulg",
"/home/aram/Manual_csv/e6fcd2b4-9621-4264-b07f-5aa4ad6df31c.ulg",

# ---------------- Stabilized ----------------
"/home/aram/Stabilized_csv/6d37fc99-a779-4752-b133-1969f71d74f5.ulg",
"/home/aram/Stabilized_csv/8e16ff5b-b3a4-4270-b9de-aa36fae1a7bd.ulg",
"/home/aram/Stabilized_csv/9ec926ae-9695-4a17-8992-d1cdd450861f.ulg",
"/home/aram/Stabilized_csv/92b39427-3d5d-4385-8aa1-de92cd250928.ulg",
"/home/aram/Stabilized_csv/633f7a4f-c446-41f9-b49f-23f0bdb79c27.ulg",
"/home/aram/Stabilized_csv/397023a6-90cc-48d8-804e-9ea15013c2a8.ulg",
"/home/aram/Stabilized_csv/916472b2-ff51-4cff-a101-c5aa30668c6f.ulg",
"/home/aram/Stabilized_csv/b9291c57-d881-4efa-b0f3-b9f2bf4c80a0.ulg",
"/home/aram/Stabilized_csv/de49af9e-87ab-46e5-844e-4e9a0137d8b7.ulg",
"/home/aram/Stabilized_csv/ef64cd3c-c6c4-4f41-8f5f-8769efe423e6.ulg",

# ---------------- Position ----------------
"/home/aram/Position_csv/5d246c38-ecaf-4bb9-9300-5e3f7aae1639.ulg",
"/home/aram/Position_csv/15fdb906-2cb5-4fcc-a0ef-3be56e827ae5.ulg",
"/home/aram/Position_csv/79d61aee-843c-4eda-9252-5cd6963e9679.ulg",
"/home/aram/Position_csv/080ad6b1-1db8-4d44-9893-77fb183ce3d2.ulg",
"/home/aram/Position_csv/958e7958-474b-4620-bb7a-bff27aa1e5a2.ulg",
"/home/aram/Position_csv/a28a066f-20a1-4a5a-b58b-4474e41149b2.ulg",
"/home/aram/Position_csv/acb018f2-3c6b-4b4a-86f8-bf72ab69e3f8.ulg",
"/home/aram/Position_csv/af7a41b8-2302-41cd-a269-4e63117184f6.ulg",
"/home/aram/Position_csv/b8e658e2-9d79-406c-ba89-f85329086749.ulg",
"/home/aram/Position_csv/f6bc8ffb-06be-4ee2-a387-5124facbfc22.ulg",

# ---------------- Altitude ----------------
"/home/aram/Altitude_csv/00c43507-ee52-4d55-9076-24b636868f51.ulg",
"/home/aram/Altitude_csv/0d3ddac4-db38-4c42-b1e9-86a8138830fd.ulg",
"/home/aram/Altitude_csv/3aac4b59-b88d-4d3b-a46f-22292a9102bf.ulg",
"/home/aram/Altitude_csv/08f6b6f3-8a36-47b7-a6c3-b2747fbcbbc0.ulg",
"/home/aram/Altitude_csv/47bae861-9cc9-4e70-9695-bb21ebdc0552.ulg",
"/home/aram/Altitude_csv/1347edcf-48a1-4fa5-a21f-61015bdd1a67.ulg",
"/home/aram/Altitude_csv/120966f7-dd3c-4336-bec0-192e60706cc5.ulg",
"/home/aram/Altitude_csv/a52751e3-535e-4f9c-906a-957cbc7d884a.ulg",
"/home/aram/Altitude_csv/cbc47116-1b93-423c-a435-f2ce025bcc47.ulg",
"/home/aram/Altitude_csv/e5b61efe-b58c-4274-bc99-7a3b7de65e39.ulg",

# ---------------- Offboard ----------------
"/home/aram/Offboard_csv/0d37c49a-3df4-4bc2-b5ec-2a000ba06c23.ulg",
"/home/aram/Offboard_csv/2ba5555c-4710-4cf0-9f9b-87a402abddcc.ulg",
"/home/aram/Offboard_csv/6e1aaf3b-2d6d-4411-a0b5-16fc698ca6ed.ulg",
"/home/aram/Offboard_csv/7a2025c2-0318-4969-898d-8244a8adabd9.ulg",
"/home/aram/Offboard_csv/69c1eca7-14da-4f95-a5b1-60fd7a6c22b3.ulg",
"/home/aram/Offboard_csv/832230a0-c977-4b10-910e-afda0b8eb0ca.ulg",
"/home/aram/Offboard_csv/a88c6081-2a82-4a67-a57e-0c5268c888a4.ulg",
"/home/aram/Offboard_csv/b9434e64-d044-4733-a46c-1c6ba23b1057.ulg",
"/home/aram/Offboard_csv/d7ddbfb1-8569-4a3a-bdb9-158e1e1c35ec.ulg",
"/home/aram/Offboard_csv/e2905b4d-ba6b-4770-92fe-ed995d0cf176.ulg",
]

OUTPUT_CSV = "merged_dataset_labeled1.csv"



# -----------------------------------
# Safe dataset getter
# -----------------------------------

def get_dataset_safe(ulog, name):
    try:
        return ulog.get_dataset(name).data
    except:
        return None


# -----------------------------------
# Extract features
# -----------------------------------

def extract_features_from_log(log_path):

    try:
        ulog = ULog(log_path)
        log_id = os.path.basename(log_path).replace(".ulg", "")

        # Label from folder name
        folder_name = os.path.basename(os.path.dirname(log_path))
        label = folder_name.replace("_csv", "")

        # ---------------- Position / Velocity ----------------
        vel = get_dataset_safe(ulog, "vehicle_local_position")

        if vel is None:
            vel = get_dataset_safe(ulog, "vehicle_odometry")

        if vel is None:
            print(f"⚠ Skipping {log_path} (no position topic)")
            return None

        vx = vel["vx"]
        vy = vel["vy"]
        vz = vel["vz"]
        z = vel["z"] if "z" in vel else np.zeros_like(vx)

        # ---------------- Acceleration ----------------
        accel = get_dataset_safe(ulog, "vehicle_acceleration")

        if accel is not None:
            ax = accel["xyz[0]"]
            ay = accel["xyz[1]"]
            az = accel["xyz[2]"]
        else:
            imu = get_dataset_safe(ulog, "sensor_combined")
            if imu is None:
                print(f"⚠ Skipping {log_path} (no accel topic)")
                return None
            ax = imu["accelerometer_m_s2[0]"]
            ay = imu["accelerometer_m_s2[1]"]
            az = imu["accelerometer_m_s2[2]"]

        # ---------------- Angular Velocity ----------------
        rates = get_dataset_safe(ulog, "vehicle_angular_velocity")

        if rates is not None:
            p = rates["xyz[0]"]
            q = rates["xyz[1]"]
            r = rates["xyz[2]"]
        else:
            imu = get_dataset_safe(ulog, "sensor_combined")
            if imu is None:
                print(f"⚠ Skipping {log_path} (no gyro topic)")
                return None
            p = imu["gyro_rad[0]"]
            q = imu["gyro_rad[1]"]
            r = imu["gyro_rad[2]"]

        # ---------------- Attitude ----------------
        att = get_dataset_safe(ulog, "vehicle_attitude")
        if att is None:
            print(f"⚠ Skipping {log_path} (no attitude topic)")
            return None

        q0 = att["q[0]"]
        q1 = att["q[1]"]
        q2 = att["q[2]"]
        q3 = att["q[3]"]

        roll = np.arctan2(
            2 * (q0 * q1 + q2 * q3),
            1 - 2 * (q1 ** 2 + q2 ** 2)
        )

        pitch = np.arcsin(
            np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0)
        )

        yaw = np.arctan2(
            2 * (q0 * q3 + q1 * q2),
            1 - 2 * (q2 ** 2 + q3 ** 2)
        )

        # ---------------- Equalize length ----------------
        min_len = min(
            len(vx), len(ax), len(p), len(roll)
        )

        df = pd.DataFrame({
            "id": [log_id] * min_len,
            "vx": vx[:min_len],
            "vy": vy[:min_len],
            "vz": vz[:min_len],
            "z": z[:min_len],
            "ax": ax[:min_len],
            "ay": ay[:min_len],
            "az": az[:min_len],
            "p": p[:min_len],
            "q": q[:min_len],
            "r": r[:min_len],
            "roll": roll[:min_len],
            "pitch": pitch[:min_len],
            "yaw": yaw[:min_len],
            "flight_mode": [label] * min_len
        })

        return df

    except Exception as e:
        print(f"❌ Error in {log_path}: {e}")
        return None


# -----------------------------------
# MAIN
# -----------------------------------

all_dfs = []

for log_file in LOG_FILES:
    print(f"Processing: {log_file}")
    df = extract_features_from_log(log_file)

    if df is not None:
        all_dfs.append(df)

if len(all_dfs) == 0:
    print("❌ No valid logs processed.")
    exit()

final_df = pd.concat(all_dfs, ignore_index=True)

final_df = final_df[
    [
        "id",
        "vx", "vy", "vz", "z",
        "ax", "ay", "az",
        "p", "q", "r",
        "roll", "pitch", "yaw",
        "flight_mode"
    ]
]

final_df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ DATASET CREATED SUCCESSFULLY")
print(f"Saved to {OUTPUT_CSV}")
print(f"Total samples: {len(final_df)}")
print(f"Classes: {final_df['flight_mode'].unique()}")
print(final_df["flight_mode"].value_counts())
flight_counts = final_df.groupby("flight_mode")["id"].nunique()
print(flight_counts)
print(final_df.isnull().sum())


#/home/aram/Downloads/merged_dataset_labeled1.csv
#/home/aram/Downloads/merged_dataset_only_augmented.csv