# 🚁 UAV Flight Mode Classification from Telemetry

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-green.svg)](https://github.com/)

## 📌 Overview
This project implements a supervised multi-class classification system to identify UAV flight modes using high-frequency onboard telemetry data. 

Flight logs were collected from public PX4 records, merged, cleaned, augmented, and transformed into structured datasets. The system identifies five specific flight modes: **Manual, Stabilized, Position, Altitude, and Offboard.**

### 🎯 Objective
* **Flight Safety Monitoring:** Real-time state detection.
* **Autonomous Mission Validation:** Ensuring the drone follows commanded logic.
* **Post-Flight Analytics:** Automated log processing for aerospace research.

---

## 📊 Dataset & Features

### 📥 Data Source
* Public UAV flight logs from the PX4 ecosystem.
* Standardized into a unified dataset.
* Split: **70% Training | 15% Validation | 15% Testing.**

### 📈 Features
* **Velocity:** $v_x, v_y, v_z$
* **Position:** $z$ (Altitude)
* **Linear Acceleration:** $a_x, a_y, a_z$
* **Angular Rates:** $p, q, r$
* **Attitude:** $Roll, Pitch, Yaw$

---

## 🧠 Data Processing Pipeline

1. **Preprocessing:** Log merging, cleaning, and label encoding.
2. **Feature Scaling:** Standardization of sensor inputs.
3. **Data Augmentation:** Sequential telemetry samples were paired and averaged to increase density while preserving temporal continuity.
4. **Balancing:** Addressed class imbalance to ensure robust performance across all flight modes.

### 🤖 Models Trained

Logistic Regression

Support Vector Machine (Linear & RBF)

Random Forest

LightGBM

K-Nearest Neighbors (KNN)

Naive Bayes

XGBoost (Best Performing Model)

---

## 📁 Repository Structure

```text
├── build_dataset.py           # Log collection and merging
├── split_dataset.py           # Train/Val/Test splitting
├── balanced_dataset.py        # Class imbalance handling
├── augment_train_only.py      # Temporal data augmentation
├── train_models.py            # Model training script
├── uav_full_system_validation.py # End-to-end evaluation
├── best_model.pkl             # Optimized XGBoost model
├── features_scaler.joblib     # Pre-trained feature scaler
└── flight_mode_encoder.joblib # Label encoder

## 🛠 Tech Stack

* **Languages:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Visualization:** Matplotlib, Seaborn
* **Model Deployment:** Joblib

---

## 💡 Key Highlights

✔ **Real-World Data:** Built using actual aerospace telemetry logs.
✔ **Classification:** Multi-class identification across 5 distinct modes.
✔ **Temporal Augmentation:** Custom strategies to increase dataset density.
✔ **Production-Ready:** Includes saved models and a full validation pipeline.
✔ **Reproducible:** Structured workflow from raw data to final inference.

---

## 🔮 Future Improvements

1.  **Temporal Deep Learning:** Integration of LSTM, GRU, or 1D-CNN architectures to better capture time-series dependencies.
2.  **Real-Time Inference:** Optimization for onboard deployment on flight controllers like Pixhawk.
3.  **Explainability:** Implementation of SHAP or LIME to visualize feature importance for flight safety audits.
4.  **Edge Deployment:** Converting models to TensorFlow Lite or ONNX for edge hardware.
5.  **Expanded Data:** Incorporating multi-airframe datasets (VTOL, Fixed-wing, and Multi-rotor).

---

## 👨‍💻 Project Context

This project was developed as a team initiative to demonstrate practical Machine Learning pipeline development within the aerospace domain. It focuses on the intersection of telemetry analytics and autonomous systems validation.

---


