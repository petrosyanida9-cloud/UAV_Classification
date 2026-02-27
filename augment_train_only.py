import pandas as pd
import numpy as np

# =========================================
# FILE PATHS
# =========================================

TRAIN_PATH = "/home/aram/Downloads/train_dataset_v2.csv"
OUTPUT_PATH = "/home/aram/Downloads/train_dataset_v2_augmented.csv"

# =========================================
# LOAD TRAIN DATA ONLY
# =========================================

df = pd.read_csv(TRAIN_PATH)

print("Original TRAIN shape:", df.shape)
print("Unique flight IDs:", df["id"].nunique())
print("Flight modes:", df["flight_mode"].unique())

# =========================================
# DEFINE FEATURE COLUMNS
# =========================================

non_feature_cols = ["id", "flight_mode"]
feature_cols = [col for col in df.columns if col not in non_feature_cols]

print("\nFeature columns used:", feature_cols)

# =========================================
# CREATE AUGMENTED SAMPLES (PAIR AVERAGING)
# =========================================

augmented_rows = []

for log_id, group in df.groupby("id"):

    group = group.reset_index(drop=True)

    if len(group) < 2:
        continue

    for i in range(0, len(group) - 1, 2):

        row1 = group.loc[i]
        row2 = group.loc[i + 1]

        averaged_features = (row1[feature_cols] + row2[feature_cols]) / 2

        new_sample = {
            "id": log_id,
            "flight_mode": row1["flight_mode"]
        }

        for col in feature_cols:
            new_sample[col] = averaged_features[col]

        augmented_rows.append(new_sample)

# =========================================
# CREATE AUGMENTED DATAFRAME
# =========================================

augmented_df = pd.DataFrame(augmented_rows)

print("Augmented TRAIN shape:", augmented_df.shape)

# =========================================
# CONCAT ORIGINAL + AUGMENTED
# =========================================

final_train_df = pd.concat([df, augmented_df], ignore_index=True)

print("Final TRAIN shape after augmentation:", final_train_df.shape)

print("\nNew label distribution:")
print(final_train_df["flight_mode"].value_counts())

# =========================================
# SAVE
# =========================================

final_train_df.to_csv(OUTPUT_PATH, index=False)

print("\n Augmented TRAIN dataset saved successfully.")
print("Saved to:", OUTPUT_PATH)

#/home/aram/Downloads/train_dataset_v2_augmented.csv