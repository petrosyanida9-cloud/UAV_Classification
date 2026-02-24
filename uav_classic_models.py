import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ==========================
# 1. Load train/val/test CSVs
# ==========================
train_df = pd.read_csv("/home/aram/Downloads/train_dataset.csv")
val_df   = pd.read_csv("/home/aram/Downloads/val_dataset.csv")
test_df  = pd.read_csv("/home/aram/Downloads/test_dataset.csv")

FEATURES = [
    'vx', 'vy', 'vz', 'z',
    'ax', 'ay', 'az',
    'p', 'q', 'r',
    'roll', 'pitch', 'yaw'
]
TARGET = "flight_mode"

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ==========================
# 2. Encode target labels
# ==========================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# Save label encoder
joblib.dump(le, "flight_mode_encoder.joblib")

# ==========================
# 3. Feature scaling (optional for tree models, mandatory for MLP)
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "features_scaler.joblib")

# ==========================
# 4. Train RandomForest
# ==========================
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train_enc)

# Evaluate RF
y_val_pred_rf = rf_clf.predict(X_val_scaled)
y_test_pred_rf = rf_clf.predict(X_test_scaled)

print("=== RANDOM FOREST ===")
print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred_rf))
print("Test Accuracy:", accuracy_score(y_test_enc, y_test_pred_rf))
print("\nClassification Report:\n", classification_report(y_test_enc, y_test_pred_rf, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_enc, y_test_pred_rf))

# Save RF model
joblib.dump(rf_clf, "RF_UAV_model.joblib")

# ==========================
# 5. Train XGBoost
# ==========================
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train_scaled, y_train_enc,
            eval_set=[(X_val_scaled, y_val_enc)],
            verbose=False)

# Evaluate XGB
y_val_pred_xgb = xgb_clf.predict(X_val_scaled)
y_test_pred_xgb = xgb_clf.predict(X_test_scaled)

print("=== XGBOOST ===")
print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred_xgb))
print("Test Accuracy:", accuracy_score(y_test_enc, y_test_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test_enc, y_test_pred_xgb, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_enc, y_test_pred_xgb))

# Save XGB model
joblib.dump(xgb_clf, "XGB_UAV_model.joblib")

print("\n Models trained and saved. You can now use them for inference.")
