# Flight Delay Prediction – Regression Analysis

This project focuses on predicting flight arrival delays using a real-world dataset.  
The goal is not to build a production model, but to demonstrate a complete machine learning workflow including preprocessing, feature engineering, model comparison, and evaluation.

---

## Problem

Predict flight arrival delay (`ARR_DELAY`) using features such as:

- Scheduled Departure Time (`CRS_DEP_TIME`)
- Scheduled Arrival Time (`CRS_ARR_TIME`)
- Distance
- Scheduled Elapsed Time (`CRS_ELAPSED_TIME`)
- Origin Airport
- Destination Airport
- Carrier

---

## Approach

The project follows a structured ML pipeline:

- Data cleaning and missing value handling  
- Categorical encoding using one-hot encoding  
- Train/test split  
- Model training and evaluation  
- Feature importance analysis  
- Dimensionality reduction using top features  

---

## Models Used

- Random Forest Regressor  
- Linear Regression  
- Dummy Regressor (baseline)  

---

## Evaluation Metrics

Models are evaluated using:

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Training Time  

Example results:

| Model              | MAE   | MSE    |
|--------------------|-------|--------|
| Random Forest      | ~23.4 | ~2257  |
| Linear Regression  | ~23.6 | ~2266  |
| Dummy Regressor    | ~23.8 | ~2300  |

---

## Results Analysis

- Random Forest slightly outperforms Linear Regression and Dummy baseline  
- Improvement over baseline is limited  
- Indicates weak predictive signal in available features  

---

## Key Insights

- Feature importance helped reduce dimensionality  
- Training time was significantly improved (from minutes to seconds)  
- Dataset lacks important real-world factors such as:
  - Weather conditions  
  - Airport congestion  
  - Operational delays  

---

## Output

The script prints:

- Model comparison table (MAE, MSE, training time)  
- Prediction comparison table (Actual vs Predicted values)  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python initial_training.py