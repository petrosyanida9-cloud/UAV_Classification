# UAV Flight Mode Classification from Telemetry

## 🚁 Project Overview
This project implements a supervised multi-class classification model to identify the flight mode of an Unmanned Aerial Vehicle (UAV). By processing high-frequency onboard telemetry data, the model mirrors real-world autopilot state estimation used in advanced flight controllers.

## 📊 The Challenge
Classifying flight modes is critical for safety monitoring and autonomous mission planning. The model must distinguish between different flight dynamics based on sensor inputs to ensure the autopilot is behaving as expected.

### 📥 Input Data (Telemetry Features)
The model analyzes the following telemetry streams:
* **Velocity:** Horizontal and Vertical components.
* **Acceleration:** 3-axis linear acceleration.
* **Attitude:** Euler angles (Roll, Pitch, Yaw).
* **Angular Rates:** Gyroscope rotation speeds.
* **Altitude:** Height above takeoff or sea level.
* **Actuator Signals:** Throttle and motor output levels.

### 🏷️ Target Labels (Flight Modes)
The system classifies data into five distinct categories:
1.  **MANUAL:** Direct pilot control.
2.  **HOVER:** Stationary position maintenance.
3.  **CRUISE:** Level flight at constant speed.
4.  **RETURN:** Autonomous flight to home location.
5.  **LAND:** Controlled vertical descent and touchdown.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn / XGBoost
* **Visualization:** Matplotlib, Seaborn
