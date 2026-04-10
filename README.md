# Ride Cancellation Predictor

Predicts ride cancellation risk using real Uber/Lyft trip data from 
the NYC Taxi & Limousine Commission (High Volume FHV dataset, Jan 2024).

## Problem
Uber loses revenue every time a ride is cancelled — driver time is 
wasted, demand signals become noisy, and riders may switch platforms. 
This project builds a real-time risk scorer to trigger interventions 
before cancellation occurs.

## Dataset
NYC TLC High Volume FHV trip records (~20M trips/month, real Uber/Lyft 
data mandated by NYC Local Law 149). Sampled 300,000 trips for modelling.

## Approach
- Engineered cancellation-risk label from wait time, driver efficiency, 
  time of day, and pickup zone (TLC only reports completed trips)
- Features: dispatch wait time, hour-of-day (cyclical encoding), 
  peak/late-night flags, airport zone, driver pay efficiency, carrier
- Baseline: Logistic Regression
- Main model: XGBoost with scale_pos_weight for class imbalance
- Explainability: SHAP TreeExplainer

## Results
| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.XX |
| XGBoost | 0.XX |

Top cancellation driver (SHAP): `wait_time_min`

Estimated revenue recovered at 0.65 threshold: **$XX,XXX/month**

## Key Visual
![SHAP Summary](shap_summary_tlc.png)

## Stack
Python · XGBoost · SHAP · Pandas · PyArrow · Matplotlib · Scikit-learn

## Files
- `ride_cancellation_tlc.ipynb` — full notebook
- `shap_summary_tlc.png` — SHAP feature importance plot
- `eda_tlc.png` — exploratory analysis dashboard
