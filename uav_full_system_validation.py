# # ============================================================
# # UAV FULL SYSTEM VALIDATION (ROBUST VERSION)
# # RF + XGB + Unknown Detection + Stability Analysis
# # ============================================================
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pyulog import ULog
# import joblib
#
# # ============================================================
# # 🔧 PATHS (Verify these exist)
# # ============================================================
#
# RF_MODEL_PATH = "/home/aram/PyCharmMiscProject/RF_UAV_model.joblib"
# XGB_MODEL_PATH = "/home/aram/PyCharmMiscProject/XGB_UAV_model.joblib"
# SCALER_PATH = "/home/aram/PyCharmMiscProject/features_scaler.joblib"
# ENCODER_PATH = "/home/aram/PyCharmMiscProject/flight_mode_encoder.joblib"
#
# TEST_FILES = {
#     "Manual_only":
#         "/home/aram/Manual_csv/d3b324da-5504-48ff-b613-e05be010063e.ulg",
#
#     "Stabilized_only":
#         "/home/aram/Stabilized_csv/6cb1b7d6-6ec3-411f-b48d-60bc2b088ce9.ulg",
#
#     "Mixed_3_modes":
#         "/home/aram/Downloads/4624b22c-2e0d-42f3-ae82-7f56645cbbf0.ulg",
#
#     "Mixed_with_Other_modes":
#         "/home/aram/Downloads/1c8fbb58-af8b-403f-9178-e12917494a8a.ulg"
# }
#
# CONFIDENCE_THRESHOLD = 0.70
#
# FEATURES = [
#     'vx','vy','vz','z',
#     'ax','ay','az',
#     'p','q','r',
#     'roll','pitch','yaw'
# ]
#
# # ============================================================
# # Quaternion → Euler
# # ============================================================
#
# def quaternion_to_euler(w, x, y, z):
#     t0 = 2.0 * (w * x + y * z)
#     t1 = 1.0 - 2.0 * (x * x + y * y)
#     roll = np.arctan2(t0, t1)
#
#     t2 = 2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     pitch = np.arcsin(t2)
#
#     t3 = 2.0 * (w * z + x * y)
#     t4 = 1.0 - 2.0 * (y * y + z * z)
#     yaw = np.arctan2(t3, t4)
#
#     return roll, pitch, yaw
#
# # ============================================================
# # SAFE FEATURE EXTRACTION
# # ============================================================
#
# def extract_features(file_path):
#
#     ulog = ULog(file_path)
#
#     def get_dataset_safe(name):
#         try:
#             return ulog.get_dataset(name).data
#         except:
#             return None
#
#     pos = get_dataset_safe('vehicle_local_position')
#     if pos is None:
#         raise RuntimeError("vehicle_local_position not found — cannot proceed.")
#
#     base_len = len(pos['vx'])
#     data = {}
#
#     data['vx'] = pos['vx']
#     data['vy'] = pos['vy']
#     data['vz'] = pos['vz']
#     data['z']  = pos['z']
#
#     # Angular velocity
#     ang = get_dataset_safe('vehicle_angular_velocity')
#     if ang is not None:
#         data['p'] = ang['x'][:base_len]
#         data['q'] = ang['y'][:base_len]
#         data['r'] = ang['z'][:base_len]
#     else:
#         print("⚠ vehicle_angular_velocity missing — filling zeros")
#         data['p'] = np.zeros(base_len)
#         data['q'] = np.zeros(base_len)
#         data['r'] = np.zeros(base_len)
#
#     # Acceleration
#     acc = get_dataset_safe('vehicle_acceleration')
#     if acc is not None:
#         data['ax'] = acc['x'][:base_len]
#         data['ay'] = acc['y'][:base_len]
#         data['az'] = acc['z'][:base_len]
#     else:
#         print("⚠ vehicle_acceleration missing — filling zeros")
#         data['ax'] = np.zeros(base_len)
#         data['ay'] = np.zeros(base_len)
#         data['az'] = np.zeros(base_len)
#
#     # Attitude
#     att = get_dataset_safe('vehicle_attitude')
#     if att is not None:
#         roll, pitch, yaw = quaternion_to_euler(
#             att['q[0]'][:base_len],
#             att['q[1]'][:base_len],
#             att['q[2]'][:base_len],
#             att['q[3]'][:base_len]
#         )
#         data['roll'] = roll
#         data['pitch'] = pitch
#         data['yaw'] = yaw
#     else:
#         print("⚠ vehicle_attitude missing — filling zeros")
#         data['roll'] = np.zeros(base_len)
#         data['pitch'] = np.zeros(base_len)
#         data['yaw'] = np.zeros(base_len)
#
#     df = pd.DataFrame(data)
#     df = df.fillna(0)
#
#     return df
#
# # ============================================================
# # Stability Analysis
# # ============================================================
#
# def analyze_transitions(predictions, model_name):
#
#     changes = np.sum(predictions[:-1] != predictions[1:])
#     total = len(predictions)
#     ratio = changes / total
#
#     print(f"\n[{model_name}] Transition Analysis")
#     print("Total samples:", total)
#     print("Mode changes:", changes)
#     print("Transition ratio:", round(ratio, 5))
#
#     if ratio < 0.01:
#         print("→ VERY STABLE")
#     elif ratio < 0.05:
#         print("→ ACCEPTABLE")
#     else:
#         print("→ UNSTABLE (Too much flickering)")
#
# # ============================================================
# # Validation
# # ============================================================
#
# def validate_file(name, path, rf_model, xgb_model, scaler, encoder):
#
#     print("\n===================================================")
#     print("Testing file:", name)
#     print("Path:", path)
#     print("===================================================")
#
#     df = extract_features(path)
#     X_scaled = scaler.transform(df[FEATURES])
#
#     # ---------------- RF ----------------
#     rf_preds_num = rf_model.predict(X_scaled)
#     rf_preds = encoder.inverse_transform(rf_preds_num)
#
#     print("\n[Random Forest] Mode Counts:")
#     print(pd.Series(rf_preds).value_counts())
#
#     analyze_transitions(rf_preds_num, "Random Forest")
#
#     # ---------------- XGB ----------------
#     xgb_preds_num = xgb_model.predict(X_scaled)
#     xgb_probs = xgb_model.predict_proba(X_scaled)
#     max_conf = np.max(xgb_probs, axis=1)
#
#     xgb_preds = encoder.inverse_transform(xgb_preds_num)
#
#     xgb_preds_with_unknown = np.where(
#         max_conf < CONFIDENCE_THRESHOLD,
#         "Unknown",
#         xgb_preds
#     )
#
#     print("\n[XGBoost] Mode Counts (with Unknown detection):")
#     print(pd.Series(xgb_preds_with_unknown).value_counts())
#
#     analyze_transitions(xgb_preds_num, "XGBoost")
#
#     # Timeline Plot
#     plt.figure(figsize=(14,4))
#     plt.plot(rf_preds_num, label="RF")
#     plt.plot(xgb_preds_num, label="XGB", alpha=0.7)
#     plt.title(f"Prediction Timeline — {name}")
#     plt.legend()
#     plt.show()
#
#     # Confidence Plot
#     plt.figure(figsize=(14,4))
#     plt.plot(max_conf)
#     plt.axhline(CONFIDENCE_THRESHOLD, linestyle='--')
#     plt.title(f"XGBoost Confidence — {name}")
#     plt.ylim(0,1)
#     plt.show()
#
# # ============================================================
# # MAIN
# # ============================================================
#
# def main():
#
#     print("Loading models...")
#     rf_model = joblib.load(RF_MODEL_PATH)
#     xgb_model = joblib.load(XGB_MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     encoder = joblib.load(ENCODER_PATH)
#
#     for name, path in TEST_FILES.items():
#         validate_file(name, path, rf_model, xgb_model, scaler, encoder)
#
#     print("\n✅ FULL VALIDATION COMPLETED")
#
# if __name__ == "__main__":
#     main()


##################################

# ============================================================
# UAV FULL SYSTEM VALIDATION (FINAL STABLE VERSION)
# Window-based | 52 Features | Best Model Only
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyulog import ULog
import joblib

# ============================================================
# 🔧 PATHS
# ============================================================

MODEL_PATH   = "/home/aram/PyCharmMiscProject/best_model.pkl"
ENCODER_PATH = "/home/aram/PyCharmMiscProject/label_encoder.pkl"

TEST_FILES = {
    "Manual_only":
        "/home/aram/Manual_csv/d3b324da-5504-48ff-b613-e05be010063e.ulg",

    "Stabilized_only":
        "/home/aram/Stabilized_csv/6cb1b7d6-6ec3-411f-b48d-60bc2b088ce9.ulg",

    "Mixed_3_modes":
        "/home/aram/Downloads/4624b22c-2e0d-42f3-ae82-7f56645cbbf0.ulg",

    "Mixed_with_Other_modes":
        "/home/aram/Downloads/1c8fbb58-af8b-403f-9178-e12917494a8a.ulg"
}

CONFIDENCE_THRESHOLD = 0.70
WINDOW_SIZE = 50
STEP = 25

FEATURE_COLUMNS = [
    "vx","vy","vz","z",
    "ax","ay","az",
    "p","q","r",
    "roll","pitch","yaw"
]

# ============================================================
# Quaternion → Euler
# ============================================================

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

# ============================================================
# Feature Extraction from ULOG
# ============================================================

def extract_features(file_path):

    ulog = ULog(file_path)
    pos = ulog.get_dataset('vehicle_local_position').data
    base_len = len(pos['vx'])

    data = {
        'vx': pos['vx'],
        'vy': pos['vy'],
        'vz': pos['vz'],
        'z':  pos['z']
    }

    try:
        ang = ulog.get_dataset('vehicle_angular_velocity').data
        data['p'] = ang['x'][:base_len]
        data['q'] = ang['y'][:base_len]
        data['r'] = ang['z'][:base_len]
    except:
        data['p'] = np.zeros(base_len)
        data['q'] = np.zeros(base_len)
        data['r'] = np.zeros(base_len)

    try:
        acc = ulog.get_dataset('vehicle_acceleration').data
        data['ax'] = acc['x'][:base_len]
        data['ay'] = acc['y'][:base_len]
        data['az'] = acc['z'][:base_len]
    except:
        data['ax'] = np.zeros(base_len)
        data['ay'] = np.zeros(base_len)
        data['az'] = np.zeros(base_len)

    try:
        att = ulog.get_dataset('vehicle_attitude').data
        roll, pitch, yaw = quaternion_to_euler(
            att['q[0]'][:base_len],
            att['q[1]'][:base_len],
            att['q[2]'][:base_len],
            att['q[3]'][:base_len]
        )
        data['roll'] = roll
        data['pitch'] = pitch
        data['yaw'] = yaw
    except:
        data['roll'] = np.zeros(base_len)
        data['pitch'] = np.zeros(base_len)
        data['yaw'] = np.zeros(base_len)

    df = pd.DataFrame(data)
    df = df.fillna(0)

    return df

# ============================================================
# Window Creation (52 Features EXACT MATCH)
# ============================================================

def create_windows(df):

    X = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        features = []

        for col in FEATURE_COLUMNS:
            values = window[col].values

            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values)
            ])

        X.append(features)

    return np.array(X)

# ============================================================
# Stability Analysis
# ============================================================

def analyze_transitions(predictions):

    changes = np.sum(predictions[:-1] != predictions[1:])
    ratio = changes / len(predictions)

    print("\nTransition ratio:", round(ratio, 5))

    if ratio < 0.01:
        print("→ VERY STABLE")
    elif ratio < 0.05:
        print("→ ACCEPTABLE")
    else:
        print("→ UNSTABLE")

# ============================================================
# Validation per file
# ============================================================

def validate_file(name, path, model, encoder):

    print("\n===================================================")
    print("Testing:", name)
    print("File:", path)
    print("===================================================")

    df = extract_features(path)
    X_windows = create_windows(df)

    if len(X_windows) == 0:
        print("Not enough samples for windowing.")
        return

    print("Total windows:", len(X_windows))

    preds_num = model.predict(X_windows)
    preds = encoder.inverse_transform(preds_num)

    print("\nMode Counts:")
    print(pd.Series(preds).value_counts())

    # Confidence analysis
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_windows)
        max_conf = np.max(probs, axis=1)

        print("\nAverage confidence:", round(np.mean(max_conf), 4))

        preds_with_unknown = np.where(
            max_conf < CONFIDENCE_THRESHOLD,
            "Unknown",
            preds
        )

        print("\nMode Counts (with Unknown detection):")
        print(pd.Series(preds_with_unknown).value_counts())

        plt.figure(figsize=(14,4))
        plt.plot(max_conf)
        plt.axhline(CONFIDENCE_THRESHOLD, linestyle='--')
        plt.title(f"Confidence — {name}")
        plt.ylim(0,1)
        plt.show()

    analyze_transitions(preds_num)

    plt.figure(figsize=(14,4))
    plt.plot(preds_num)
    plt.title(f"Prediction Timeline — {name}")
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("Model type:", type(model))
    print("Expected features:", model.n_features_in_)

    for name, path in TEST_FILES.items():
        validate_file(name, path, model, encoder)

    print("\nFULL VALIDATION COMPLETED")

if __name__ == "__main__":
    main()