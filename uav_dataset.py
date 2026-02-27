import pandas as pd

files = [
    "Manual_dataset.csv",
    "Stabilized_dataset.csv",
    "Position_dataset.csv",
    "Altitude_dataset.csv",
    "Offboard_dataset.csv"
]

df_list = [pd.read_csv(f"/home/aram/Downloads/{f}") for f in files]

full_df = pd.concat(df_list, ignore_index=True)

full_df.to_csv("/home/aram/Downloads/full_dataset.csv", index=False)

print("Full dataset shape:", full_df.shape)
print(full_df["flight_mode"].value_counts())