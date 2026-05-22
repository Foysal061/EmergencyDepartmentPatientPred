# EmergencyDepartmentPatientPred

Hourly Emergency Department (ED) patient arrival forecasting at Vestfold Hospital Trust (Sykehuset i Vestfold, SiV; Tønsberg, Norway). This repository contains the experimental notebooks used in the master's thesis *Forecast Capacity and Influx of Patients to the Emergency Department*, which evaluates five machine learning models against the hospital's existing calendar-based Median Baseline across various horizons. The underlying dataset is the property of Vestfold Hospital Trust and is not distributed here; see [Data](#data) for what the notebooks expect.

## Live demo

A prototype inference API for the strongest long-horizon model (Direct LSTM+Optuna) is published on Hugging Face Spaces:

https://huggingface.co/spaces/Foysal061/ED_arrival_LSTM

The Space accepts 24 hourly rows of 46 engineered features and returns 24 hourly arrival counts, matching the 24-hour horizon of the existing Vestfold planning dashboard.

## Models

| Notebook | Model | Description |
|---|---|---|
| `MedianEdArrival.ipynb` | Median Baseline | Calendar lookup (7 × 24 grid of medians), the hospital's existing operational system |
| `LSTM.ipynb` | LSTM | Manually tuned recurrent network with autoregressive multi-step rollout |
| `BiDirectional_Lstm.ipynb` | BiLSTM | Bidirectional LSTM with the same training protocol as the LSTM |
| `LSTMWithOptuna.ipynb` | LSTM + Optuna | LSTM with TPE-based hyperparameter search |
| `LightGBM.ipynb` | LightGBM | Gradient boosting with temporal features recomputed at every rollout step |
| `Direct24hrLSTMWithOptuna.ipynb` | Direct LSTM + Optuna | Multi-output LSTM that predicts all 24 future hours in a single forward pass; this is the model deployed on Hugging Face |

### Utility notebook

| Notebook | Description |
|---|---|
| `HFInputCSVGenerator.ipynb` | Produces a 24-row × 46-feature CSV in the exact column order the deployed Hugging Face Space expects. Pulls hourly weather observations for the Vestfold coordinates from the Open-Meteo Archive API, fetches Norwegian public holidays from the Nager.Date API, reads monthly infection totals from `Infeksjonsdata.xlsx` (private, see [Data](#data)), and computes the engineered calendar, lag, EMA, and triage features. The resulting file can be uploaded directly to the **Upload CSV** tab of the Space without any manual feature engineering. |

## Data

The dataset that backs every notebook in this repository is the
property of Vestfold Hospital Trust (SiV) and **cannot be redistributed
here**. Anyone wishing to reproduce the experiments must obtain the
two source files directly from the hospital, under a separate data
agreement, and place them in the working directory before running the
notebooks. The notebooks expect:

| File (not included) | Description |
|---|---|
| `VestfoldTriageReport.csv` | 12 months of de-identified ED visit records from Vestfold Hospital Trust (October 2023 to October 2024, 35,674 patient visits). Each row records arrival time, departure time, first doctor response time, and first triage time. |
| `Infeksjonsdata.xlsx` | Monthly total infection counts as supplied by Vestfold Hospital Trust. One row per month with two columns: `Month` (e.g. `Oct-23`) and `Total_Infected_Patient_Monthly`. |

Two of the three external inputs the notebooks use are public and are
fetched at runtime, so nothing further has to be downloaded by hand:
hourly weather observations (temperature, humidity, precipitation,
surface pressure, wind speed) come from the Open-Meteo Archive API for
the Vestfold Hospital coordinates, and Norwegian public holidays come
from the Nager.Date API.

## Method

- **Train/validation/test split.** Chronological 80/10/10. The Median Baseline uses a separate seven-day evaluation window (18-25 October 2024) because it is a statistical lookup with no training phase.
- **Feature set.** 46 engineered features per hour, including calendar variables, lag features, exponential moving averages, weather, monthly infection rate, holiday indicators, and triage-category distributions.
- **Evaluation horizons.** 1, 2, 6, 12, and 24 hours.
- **Metrics.** RMSE, MAE, R², MAPE.

## Results

R² on the held-out test set:

| Model | 1h | 6h | 12h | 24h |
|---|---|---|---|---|
| LSTM | 0.355 | 0.208 | -0.360 | 0.031 |
| BiLSTM | 0.382 | 0.219 | -0.084 | 0.028 |
| LSTM+Optuna | 0.398 | 0.213 | -0.454 | 0.075 |
| LightGBM | 0.460 | 0.288 | 0.137 | 0.098 |
| **Direct LSTM+Optuna** | 0.291 | 0.290 | **0.255** | **0.176** |

The Median Baseline reaches R² = 0.025 on its own seven-day window.

**Per-horizon recommendation.** LSTM+Optuna at 1-2 hours (lowest RMSE), BiLSTM at 6 hours, and Direct LSTM+Optuna at 12-24 hours (the only model that maintains a positive R² across the full horizon range). The autoregressive LSTM variants collapse to a negative R² at 12 hours because their cyclical temporal features stay frozen during recursive rollout; LightGBM avoids this by recomputing features at every step, and the Direct LSTM+Optuna sidesteps the issue by removing rollout altogether.

## How to run

The notebooks were developed in Google Colab with data stored on Google Drive. Running them in Colab is the easiest option; a local setup is also supported.

### Option A: Google Colab (recommended)

1. Obtain the two data files (`VestfoldTriageReport.csv` and `Infeksjonsdata.xlsx`) from Vestfold Hospital Trust (see the [Data](#data) section above) and upload them to a folder in your Google Drive, for example `MyDrive/Colab Notebooks/`.

2. Open a notebook in Colab. Either click the notebook on GitHub and use the **Open in Colab** badge, or go to:

   https://colab.research.google.com/github/Foysal061/EmergencyDepartmentPatientPred

   and pick the notebook you want to run.

3. Install the Python packages that Colab does not ship with by adding (or uncommenting) this line at the top of the first code cell:

   ```python
   !pip install openmeteo-requests requests-cache retry-requests optuna
   ```

   `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, and `seaborn` are already available on Colab and do not need to be installed.

4. Mount your Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. In the configuration cell near the top of the notebook, set `ED_DATA_PATH` and `INFECTION_DATA_PATH` to point to the files you uploaded in step 1, for example:

   ```python
   ED_DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/VestfoldTriageReport.csv"
   INFECTION_DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/Infeksjonsdata.xlsx"
   ```

6. Run all cells (`Runtime > Run all`).

### Option B: Local Jupyter

1. Clone the repository:

   ```bash
   git clone https://github.com/Foysal061/EmergencyDepartmentPatientPred.git
   cd EmergencyDepartmentPatientPred
   ```

2. Install the dependencies:

   ```bash
   pip install openmeteo-requests requests-cache retry-requests numpy pandas tensorflow scikit-learn optuna lightgbm matplotlib seaborn
   ```

3. Obtain the two data files from Vestfold Hospital Trust (see the [Data](#data) section above) and place them in the working directory. Open a notebook in Jupyter or VS Code, and in the configuration cell replace the `path/to/...` placeholders with the local paths, for example `./VestfoldTriageReport.csv` and `./Infeksjonsdata.xlsx`. The Drive-mount cell can be commented out or removed when running locally.

4. Run the cells from top to bottom.

## Repository structure

```
EmergencyDepartmentPatientPred/
├── BiDirectional_Lstm.ipynb        # BiLSTM experiment
├── Direct24hrLSTMWithOptuna.ipynb  # Direct multi-output LSTM (deployed model)
├── HFInputCSVGenerator.ipynb       # Generates a CSV input for the Hugging Face Space
├── LSTM.ipynb                      # Manually tuned LSTM
├── LSTMWithOptuna.ipynb            # LSTM with Optuna hyperparameter search
├── LightGBM.ipynb                  # LightGBM gradient boosting
├── MedianEdArrival.ipynb           # Median Baseline (operational reference)
└── README.md
```

## Thesis

This codebase accompanies the master's thesis *Forecast Capacity and Influx of Patients to the Emergency Department*, completed at the University of South-Eastern Norway (USN). The thesis describes the methodology, experimental design, results, and discussion in full detail.
