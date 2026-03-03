# Virtual Vahana 2026 – Phase 1: ADAS Fusion System

**Institution:** SRM Institute of Science and Technology, Kattankulathur
**Competition:** Virtual Vahana 2026 (Phase 1: Safety-first ADAS + Assisted Driving Systems)  
**Simulator:** CARLA 0.9.x  

---

## 👥 Team Details

| Name  |  
| :--- | 
| **1. Karan Rajesh** | 
| **2. Vaikunth MS** | 
| **3. Tharun Kumaar RK** | 
| **4. Mahesh Paul J** | 
| **5. Sagar Sewak** |
| **6. Francis Solomon X** |
| **7. Harishankar MN** |

---

## 🚀 Project Overview

This repository contains our Phase 1 submission for the Virtual Vahana 2026 competition. We have developed a "safety-first" Advanced Driver-Assistance System (ADAS) using the CARLA Python API. 

Our system operates on a Level 2 autonomy paradigm where active safety overrides convenience. The core architecture fuses data from Semantic LiDAR, Raycast, and Radar sensors to provide reliable environmental awareness while filtering out "ghost" readings. 

**Key Features Implemented:**
* **AEB & FCW:** 3-Stage Autonomous Emergency Braking with Time-To-Collision (TTC) and kinematic threshold logic. Absolute priority override.
* **ACC:** Adaptive Cruise Control with PID longitudinal tracking and Traffic Light/Sign Recognition (TSR) compliance.
* **LKA & LDW:** Lane Keeping Assist with speed-adaptive authority, rate-limiting, and IIR smoothing for highway stability.
* **BSM (Innovation):** Dual-sided Blind Spot Monitoring with lateral collision risk alerts.
* **HMI:** Custom PyGame UI featuring a Tesla-style spatial mini-map, telemetry panel, and steering contribution breakdown.

---

## 📂 Repository Structure & Deliverables Mapping

All mandatory deliverables for Stage 1 evaluation are included in this repository:

1. **`Demo_Video/`** - Contains the live recorded simulator demo (`.mp4`) showing all ADAS features in action. *(If the file is too large for GitHub, a public Google Drive link is provided in a text file here).*
2. **`Architecture/`** - Contains the high-resolution ADAS System Architecture Diagram (`.png` / `.pdf`).
3. **`Documentation/Feature_Logic_Explanation.pdf`** - Detailed breakdown of TTC calculations, thresholds, and arbitration logic.
4. **`Dashboard_Media/`** - Screenshots and short clips of the PyGame HMI UI, alerts, and spatial map.
5. **`Documentation/Technical_Report.pdf`** - Our comprehensive 8–12 page technical report.
6. **`combined_adas_merged.py`** - **[MAIN SCRIPT]** The executable Python script containing our entire sensor fusion and ADAS logic.

---

## ⚙️ Steps to Run the Simulator

### Prerequisites
* **CARLA Simulator:** Version 0.9.x (Tested on 0.9.11)
* **Python:** 3.7 to 3.12
* **Required Libraries:** `pygame`, `numpy`, `carla`

### Execution Instructions
1. **Start the CARLA Server:**
   Launch the CARLA simulator executable on your machine.
   * *Windows:* Run `CarlaUE4.exe`
   * *Linux:* Run `./CarlaUE4.sh`
   
2. **Setup the Environment:**
   Open a terminal in the root directory of this repository and install the required dependencies:
   ```bash
   pip install pygame numpy carla
    ```

3. **Run the ADAS Client:**
    Ensure the CARLA server is fully loaded, then execute the main script:
    ```bash
    python adas.py
    ```



### 🎮 Keyboard Controls (For Judges & Testers)

**Driving (Manual Control Mandatory):**

* `W` / `S` : Throttle / Brake
* `A` / `D` : Steer Left / Right
* `Q` : Toggle Reverse Gear

**ADAS Feature Toggles:**

* `.` (Period) : **Master AEB Switch** (Toggles Active Safety ON/OFF)
* `J` : **ADAS Master Switch** (Enables ACC / FCW readiness)
* `Y` : **Toggle ACC** (Activates Adaptive Cruise Control)
* `U` / `O` : Increase / Decrease ACC Target Speed (+/- 5 km/h)
* `K` : **Cycle LKA/LDW** (BOTH ON -> LKA OFF -> BOTH OFF -> BOTH ON)
* `F2` : Toggle Right-Side ADAS Telemetry Panel
* `T` : Toggle Tesla-style Spatial Mini-map

---

*Built with precision for Virtual Vahana 2026. Safety first, autonomy next.*