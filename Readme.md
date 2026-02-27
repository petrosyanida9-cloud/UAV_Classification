🚁 UAV Flight Mode Classification from Telemetry
📌 Overview

This project implements a supervised multi-class classification system to identify UAV flight modes using high-frequency onboard telemetry data.

Flight logs were collected from public records of PX4 via PX4 Flight Review, merged, cleaned, augmented, and transformed into structured datasets for machine learning.

The system classifies UAV telemetry into five flight modes:

Manual

Stabilized

Position

Altitude

Offboard

This project was developed as a team initiative and serves as an internship-ready aerospace machine learning portfolio project.

🎯 Objective

Correct identification of UAV flight modes is critical for:

Flight safety monitoring

Autonomous mission validation

Autopilot state verification

Post-flight analytics

The goal was to build a robust and reproducible ML pipeline capable of distinguishing flight dynamics purely from onboard telemetry signals.

📊 Dataset
📥 Data Source

Public UAV flight logs from PX4 ecosystem

Logs combined from multiple real flights

Cleaned and standardized into a unified dataset

🔢 Dataset Split

80% — Training

10% — Validation

10% — Testing

Final performance was evaluated on the independent 10% test set.

📈 Features Used
Velocity

vx, vy, vz

Position

z

Linear Acceleration

ax, ay, az

Angular Rates

p, q, r

Attitude

roll, pitch, yaw

Target Variable:
flight_mode

🧠 Data Processing Pipeline

The project includes a structured preprocessing workflow:

Log collection and merging

Data cleaning and formatting

Feature scaling

Label encoding

Dataset balancing

Data augmentation

Train / Validation / Test split

🔄 Data Augmentation Strategy

To improve generalization:

Sequential telemetry samples were paired

The mean of two consecutive samples was computed

The averaged sample was appended to the training dataset

This preserves temporal continuity while increasing dataset density.

⚖ Dataset Balancing

Balanced models were trained to address class imbalance, improving robustness across flight modes.

🤖 Models Trained

Multiple classical ML models were trained and saved:

Logistic Regression

Support Vector Machine (Linear & RBF)

Random Forest

LightGBM

XGBoost

Balanced versions of models were also trained.

🏆 Best Performing Model

XGBoost achieved the best validation performance and was selected as the final model.

(Specific F1-score will be added.)

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

A full-system validation script is included for end-to-end evaluation.

📁 Repository Structure
├── Readme.md
├── dataframe_creation.md
├── final_uav_telemetry.csv
│
├── build_dataset.py
├── split_dataset.py
├── balanced_dataset.py
├── augment_train_only.py
├── create_window_dataset.py
├── flight_level_dataset.py
│
├── uav_dataset.py
├── train_models.py
├── uav_classic_models.py
├── uav_full_system_validation.py
├── test.py
│
├── best_model.pkl
├── XGBoost_model_v2.pkl
├── XGBoost_balanced_model.joblib
├── LightGBM_balanced_model.joblib
├── LogisticRegression_balanced_model.joblib
├── SVM_balanced_model.joblib
├── SVM_RBF_model_v2.pkl
│
├── features_scaler.joblib
├── uav_scaler_v2.pkl
└── flight_mode_encoder.joblib
🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Build / Prepare Dataset
python build_dataset.py
python split_dataset.py
3️⃣ Train Models
python train_models.py
4️⃣ Run Evaluation
python uav_full_system_validation.py

Or test directly:

python test.py
🛠 Tech Stack

Python 3.x

Pandas

NumPy

Scikit-Learn

XGBoost

LightGBM

Matplotlib

Seaborn

Joblib

💡 Key Highlights

✔ Real-world aerospace telemetry dataset
✔ Multi-class classification (5 flight modes)
✔ Custom temporal data augmentation
✔ Balanced dataset training
✔ Multiple model benchmarking
✔ Saved production-ready models
✔ End-to-end validation pipeline
✔ Reproducible ML workflow

🔮 Future Improvements

Temporal deep learning models (LSTM / GRU / CNN)

Real-time onboard inference

Feature importance & explainability (SHAP)

Deployment as an API

Edge deployment on flight hardware

Expansion to larger multi-airframe datasets

👨‍💻 Project Context

Developed as a team machine learning project focused on UAV telemetry analytics and applied classification systems.
Designed to demonstrate practical ML pipeline development using real aerospace flight data.
