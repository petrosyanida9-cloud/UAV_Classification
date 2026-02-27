import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================
WINDOW_SIZE = 50
STEP = 25

FEATURES = [
    'vx','vy','vz','z',
    'ax','ay','az',
    'p','q','r',
    'roll','pitch','yaw'
]

TARGET = "flight_mode"
FLIGHT_COLUMN = "id"

#TRAIN_PATH = "/home/aram/Downloads/train_dataset_v2.csv"
TRAIN_PATH = "/home/aram/Downloads/train_dataset_v2_augmented.csv"
VAL_PATH   = "/home/aram/Downloads/val_dataset_v2.csv"
TEST_PATH  = "/home/aram/Downloads/test_dataset_v2.csv"


# ==========================
# Window Creator
# ==========================
def create_windows(df, dataset_name="dataset"):

    window_data = []

    for flight_id in df[FLIGHT_COLUMN].unique():

        flight_df = df[df[FLIGHT_COLUMN] == flight_id].reset_index(drop=True)

        if len(flight_df) < WINDOW_SIZE:
            continue

        for start in range(0, len(flight_df) - WINDOW_SIZE + 1, STEP):

            window = flight_df.iloc[start:start+WINDOW_SIZE]

            features = []

            for col in FEATURES:
                features.append(window[col].mean())
                features.append(window[col].std())
                features.append(window[col].min())
                features.append(window[col].max())

            label = window[TARGET].mode()[0]

            window_data.append(features + [label, flight_id])

    # Create column names
    columns = []
    for col in FEATURES:
        columns += [
            f"{col}_mean",
            f"{col}_std",
            f"{col}_min",
            f"{col}_max"
        ]

    columns += [TARGET, FLIGHT_COLUMN]

    window_df = pd.DataFrame(window_data, columns=columns)

    print(f"✅ {dataset_name} windows created: {len(window_df)} samples")

    return window_df


# ==========================
# MAIN
# ==========================
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Creating train windows...")
train_win = create_windows(train_df, "TRAIN")

print("Creating val windows...")
val_win = create_windows(val_df, "VAL")

print("Creating test windows...")
test_win = create_windows(test_df, "TEST")

train_win.to_csv("train_window_augmented.csv", index=False)
val_win.to_csv("val_window.csv", index=False)
test_win.to_csv("test_window.csv", index=False)

print("\n Window datasets created successfully.")
print("Train windows:", len(train_win))
print("Val windows:", len(val_win))
print("Test windows:", len(test_win))

#/home/aram/Downloads/train_window_augmented.csv