# import pandas as pd
# import numpy as np
# import joblib
# import time
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.preprocessing import LabelEncoder
#
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
#
# # =====================================
# # 1. LOAD TRAIN + VAL DATA (ALREADY SCALED)
# # =====================================
#
# train_df = pd.read_csv("/home/aram/Downloads/train_dataset_v2.csv")
# val_df   = pd.read_csv("/home/aram/Downloads/val_dataset_v2.csv")
#
# FEATURES = [
#     'vx', 'vy', 'vz', 'z',
#     'ax', 'ay', 'az',
#     'p', 'q', 'r',
#     'roll', 'pitch', 'yaw'
# ]
#
# LABEL = "flight_mode"
#
# X_train = train_df[FEATURES].values
# y_train = train_df[LABEL].values
#
# X_val = val_df[FEATURES].values
# y_val = val_df[LABEL].values
#
# # =====================================
# # 2. LABEL ENCODING
# # =====================================
#
# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)
# y_val_enc   = le.transform(y_val)
#
# joblib.dump(le, "/home/aram/Downloads/label_encoder_v2.pkl")
#
# print("Classes:", le.classes_)
#
# # =====================================
# # 3. DEFINE MODELS (Moderate but strong)
# # =====================================
#
# models = {
#
#     "RandomForest": RandomForestClassifier(
#         n_estimators=250,
#         max_depth=20,
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1
#     ),
#
#     "LogisticRegression": LogisticRegression(
#         max_iter=1000,
#         class_weight="balanced",
#         n_jobs=-1
#     ),
#
#     "SVM_RBF": SVC(
#         kernel="rbf",
#         C=5,
#         gamma="scale",
#         class_weight="balanced"
#     ),
#
#     "XGBoost": XGBClassifier(
#         n_estimators=300,
#         max_depth=6,
#         learning_rate=0.1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         eval_metric="mlogloss",
#         random_state=42,
#         n_jobs=-1
#     ),
#
#     "LightGBM": LGBMClassifier(
#         n_estimators=300,
#         learning_rate=0.1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1
#     )
# }
#
# # =====================================
# # 4. TRAIN + VALIDATE
# # =====================================
#
# results = []
#
# for name, model in models.items():
#
#     print(f"\n============================")
#     print(f"Training {name}")
#     print(f"============================")
#
#     start = time.time()
#     model.fit(X_train, y_train_enc)
#     train_time = time.time() - start
#
#     y_pred = model.predict(X_val)
#
#     acc = accuracy_score(y_val_enc, y_pred)
#     f1  = f1_score(y_val_enc, y_pred, average="macro")
#
#     print(f"Validation Accuracy: {acc:.4f}")
#     print(f"Validation Macro-F1: {f1:.4f}")
#
#     print("\nClassification Report:")
#     print(classification_report(y_val_enc, y_pred, target_names=le.classes_))
#
#     joblib.dump(model, f"/home/aram/Downloads/{name}_model_v2.pkl")
#
#     results.append([name, acc, f1, train_time])
#
# # =====================================
# # 5. COMPARISON TABLE
# # =====================================
#
# results_df = pd.DataFrame(
#     results,
#     columns=["Model", "Val Accuracy", "Val Macro-F1", "Train Time (sec)"]
# )
#
# results_df = results_df.sort_values(by="Val Macro-F1", ascending=False)
#
# print("\n===================================")
# print(" MODEL COMPARISON (VALIDATION SET)")
# print("===================================")
# print(results_df)
#
# results_df.to_csv("/home/aram/Downloads/model_comparison_v2.csv", index=False)
#
# print("\n✅ Training complete.")
# print("Now we will choose top models and evaluate on TEST.")

#######################################

# import pandas as pd
# import numpy as np
# import joblib
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
#
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
# from sklearn.utils.class_weight import compute_class_weight
#
# # ==========================
# # 1. Load datasets
# # ==========================
# train_df = pd.read_csv("/home/aram/Downloads/train_dataset_v2.csv")
# val_df   = pd.read_csv("/home/aram/Downloads/val_dataset_v2.csv")
# test_df  = pd.read_csv("/home/aram/Downloads/test_dataset_v2.csv")
#
# FEATURES = [
#     'vx','vy','vz','z',
#     'ax','ay','az',
#     'p','q','r',
#     'roll','pitch','yaw'
# ]
#
# TARGET = "flight_mode"
#
# X_train = train_df[FEATURES]
# y_train = train_df[TARGET]
#
# X_val = val_df[FEATURES]
# y_val = val_df[TARGET]
#
# X_test = test_df[FEATURES]
# y_test = test_df[TARGET]
#
# # ==========================
# # 2. Encode labels
# # ==========================
# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)
# y_val_enc   = le.transform(y_val)
# y_test_enc  = le.transform(y_test)
#
# joblib.dump(le, "flight_mode_encoder.joblib")
#
# # ==========================
# # 3. Compute class weights
# # ==========================
# classes = np.unique(y_train_enc)
# weights = compute_class_weight(
#     class_weight='balanced',
#     classes=classes,
#     y=y_train_enc
# )
#
# class_weights = dict(zip(classes, weights))
# print("Class weights:", class_weights)
#
# # ==========================
# # 4. Models (Balanced + Moderate Complexity)
# # ==========================
#
# models = {
#
#     "RandomForest": RandomForestClassifier(
#         n_estimators=300,
#         max_depth=14,
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1
#     ),
#
#     "LogisticRegression": LogisticRegression(
#         max_iter=1000,
#         class_weight="balanced",
#         n_jobs=-1
#     ),
#
#     "SVM": SVC(
#         kernel='rbf',
#         class_weight="balanced",
#         probability=False
#     ),
#
#     "LightGBM": LGBMClassifier(
#         n_estimators=300,
#         learning_rate=0.05,
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1
#     ),
#
#     "XGBoost": XGBClassifier(
#         n_estimators=300,
#         max_depth=6,
#         learning_rate=0.05,
#         objective="multi:softprob",
#         eval_metric="mlogloss",
#         random_state=42,
#         n_jobs=-1
#     )
# }
#
# results = []
#
# # ==========================
# # 5. Train & Evaluate
# # ==========================
#
# for name, model in models.items():
#     print(f"\n========== {name} ==========")
#
#     model.fit(X_train, y_train_enc)
#
#     y_pred = model.predict(X_test)
#
#     acc = accuracy_score(y_test_enc, y_pred)
#     macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
#     weighted_f1 = f1_score(y_test_enc, y_pred, average='weighted')
#
#     print("Accuracy:", acc)
#     print("Macro-F1:", macro_f1)
#     print("Weighted-F1:", weighted_f1)
#     print("\nClassification Report:\n",
#           classification_report(y_test_enc, y_pred, target_names=le.classes_))
#
#     print("\nConfusion Matrix:\n",
#           confusion_matrix(y_test_enc, y_pred))
#
#     results.append([name, acc, macro_f1, weighted_f1])
#
#     joblib.dump(model, f"{name}_balanced_model.joblib")
#
# # ==========================
# # 6. Save comparison
# # ==========================
# results_df = pd.DataFrame(
#     results,
#     columns=["Model","Accuracy","Macro-F1","Weighted-F1"]
# )
#
# results_df = results_df.sort_values(by="Macro-F1", ascending=False)
# results_df.to_csv("model_comparison_balanced.csv", index=False)
#
# print("\nFinal Ranking:")
# print(results_df)
#
# print("\n✅ All models trained and saved.")


####################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# LOAD DATA
# ==========================
train_df = pd.read_csv("/home/aram/Downloads/train_window_augmented.csv")
#train_df = pd.read_csv("/home/aram/Downloads/train_window.csv")
val_df   = pd.read_csv("/home/aram/Downloads/val_window.csv")
test_df  = pd.read_csv("/home/aram/Downloads/test_window.csv")

TARGET = "flight_mode"

# Drop id column
X_train = train_df.drop(columns=[TARGET, "id"])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET, "id"])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET, "id"])
y_test = test_df[TARGET]

feature_names = X_train.columns

# ==========================
# ENCODE LABELS
# ==========================
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val   = le.transform(y_val)
y_test  = le.transform(y_test)

class_names = le.classes_

# ==========================
# CLASS WEIGHTS
# ==========================
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# ==========================
# MODELS
# ==========================
models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist"
    ),

    "CatBoost": CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.03,
        auto_class_weights="Balanced",
        verbose=0
    )
}

# ==========================
# TRAIN + EVALUATE
# ==========================
results = {}

for name, model in models.items():

    print(f"\n==============================")
    print(f"Training {name}")
    print("==============================")

    model.fit(X_train, y_train)

    val_preds  = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_acc  = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)

    val_f1  = f1_score(y_val, val_preds, average="weighted")
    test_f1 = f1_score(y_test, test_preds, average="weighted")

    print(f"VAL Accuracy:  {val_acc}")
    print(f"TEST Accuracy: {test_acc}")
    print(f"VAL F1:  {val_f1}")
    print(f"TEST F1: {test_f1}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=class_names))

    results[name] = test_acc

# ==========================
# SELECT BEST MODEL
# ==========================
best_model_name = max(results, key=results.get)
print("\n🏆 Best model:", best_model_name)

best_model = models[best_model_name]

# ==========================
# FINAL TEST PREDICTIONS
# ==========================
final_test_preds = best_model.predict(X_test)

# ==========================
# CONFUSION MATRIX
# ==========================
cm = confusion_matrix(y_test, final_test_preds)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(f"Confusion Matrix - {best_model_name} (Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ==========================
# FEATURE IMPORTANCE (if available)
# ==========================
if hasattr(best_model, "feature_importances_"):

    importances = best_model.feature_importances_

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title(f"Feature Importance - {best_model_name}")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# ==========================
# SAVE BEST MODEL
# ==========================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Best model saved.")