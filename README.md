# Passive Walker Dynamics with Echo State Networks (ESN)

This project explores **reservoir computing (ESN)** for modeling bifurcations and impact dynamics in a passive dynamic walker (PDW).  
It includes Python scripts and Jupyter notebooks for training, prediction, and visualization.


## 📂 Project Structure
```
pwd_project/
├── real_data                      # Folder contains real PDW data
├── esn_runner.py                  # ESN core implementation
├── run_prediction_gamma_train.py  # Run predictions for fixed gamma training
├── multi-attractor.py             # Run multi-attractor test
├── unseen_attractor.py            # Run unseen attractor reconstruction test
├── real_bifurcation.csv           # Reference bifurcation data
├── requirements.txt               # Dependencies
├── LICENSE                        # Open-source license
└── README.md                      # Project description
```

## 🚀 Installation
Clone this repository:
```bash
git clone https://github.com/JonyeeShen/Passive_Dynamic_Walking_Emulation_by_Reservoir_Computing.git
cd Passive_Dynamic_Walking_Emulation_by_Reservoir_Computing
```

Install dependencies:
```bash
pip install -r requirements.txt
```
