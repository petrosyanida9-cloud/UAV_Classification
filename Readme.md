🚁 UAV Flight Mode Classification from Telemetry

📌 Project Overview

This project implements a supervised multi-class classification system to identify UAV flight modes using high-frequency onboard telemetry data.

By analyzing velocity, acceleration, angular rates, and attitude signals, the system mirrors real-world autopilot state estimation used in modern flight controllers.

Telemetry logs were collected from public flight records of PX4 via PX4 Flight Review, researched, merged, cleaned, and transformed into a structured ML-ready dataset.

This project was developed as a team initiative and serves as a portfolio-ready aerospace machine learning project.

🎯 Objective

The goal is to classify UAV telemetry data into one of five flight modes:

Manual

Stabilized

Position

Altitude

Offboard

Accurate classification of flight modes is critical for:

🛡️ Safety monitoring

🤖 Autonomous mission validation

📊 Post-flight analysis

🧠 Autopilot behavior verification

📊 Dataset
📥 Data Source

Public UAV flight logs from the PX4 ecosystem

Logs combined from multiple real flights

Cleaned and standardized into a unified dataset

🔢 Dataset Split

80% — Training

10% — Validation

10% — Testing

The final evaluation was performed on the held-out 10% test dataset.

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

🎯 Target Variable

flight_mode

🧠 Data Augmentation Strategy

To improve generalization and increase dataset diversity:

Sequential telemetry samples were paired

The mean of two consecutive data points was computed

The averaged sample was appended to the dataset

This approach:

Preserves temporal continuity

Increases effective dataset size

Avoids artificial noise injection

🛠 Tech Stack

Language: Python 3.x

Data Handling: Pandas, NumPy

Machine Learning: Scikit-Learn, XGBoost

Visualization: Matplotlib, Seaborn

Model Persistence: Joblib

🤖 Models Evaluated

We trained and compared multiple classical machine learning models:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Gradient Boosting

XGBoost

🏆 Best Performing Model: XGBoost

XGBoost achieved the highest validation performance and was selected as the final model.

🔹 Final F1-score and detailed metrics will be added.

📊 Evaluation Metrics

Models were evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Evaluation was performed on a separate validation set and confirmed on the independent test set.

📁 Repository Structure
├── Readme.md
├── dataframe_creation.md
├── final_uav_telemetry.csv
├── uav_dataset.py
├── uav_classic_models.py
├── test.py
├── features_scaler.joblib
└── flight_mode_encoder.joblib
📄 File Description

uav_dataset.py → Dataset preprocessing & feature preparation

uav_classic_models.py → Model training & benchmarking

test.py → Model evaluation on test dataset

features_scaler.joblib → Saved feature scaler

flight_mode_encoder.joblib → Encoded label transformer

dataframe_creation.md → Documentation of dataset construction process

🚀 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Train Models
python uav_classic_models.py
4️⃣ Run Testing
python test.py
📌 Key Highlights

✔ Multi-class UAV flight mode classification
✔ Real-world PX4 telemetry data
✔ Custom temporal data augmentation
✔ Multiple model benchmarking
✔ Structured Train/Validation/Test split
✔ Reproducible ML pipeline
✔ Saved scaler and encoder for deployment

🔍 Why This Project Is Internship-Ready

Uses real aerospace telemetry data

Demonstrates full ML lifecycle (data collection → preprocessing → training → evaluation)

Applies model benchmarking and validation strategy

Implements reproducible pipeline with saved artifacts

Shows practical understanding of classification in control systems

🔮 Future Improvements

LSTM / Temporal Deep Learning models

Real-time flight mode inference

Feature importance & interpretability analysis

Deployment-ready inference API

Edge deployment on embedded flight hardware

Larger multi-airframe dataset expansion
