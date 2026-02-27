import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("/home/aram/PyCharmMiscProject/merged_dataset_labeled1.csv")

# find smallest class size
min_count = df["flight_mode"].value_counts().min()

balanced_df = []

for mode in df["flight_mode"].unique():
    df_mode = df[df["flight_mode"] == mode]
    df_downsampled = resample(
        df_mode,
        replace=False,
        n_samples=min_count,
        random_state=42
    )
    balanced_df.append(df_downsampled)

balanced_df = pd.concat(balanced_df)

balanced_df.to_csv("/home/aram/Downloads/balanced_dataset1.csv", index=False)

print(balanced_df["flight_mode"].value_counts())