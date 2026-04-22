# F1 Podium Predictor

A machine learning web app that predicts Formula 1 podium finishes using a Random Forest Classifier trained on 70+ years of F1 race data.


## 📁 Project Structure

```
f1_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
│── (optional — add for real predictions)
├── results.csv          ← From Kaggle F1 dataset
├── races.csv
├── drivers.csv
├── constructors.csv
└── circuits.csv
```

## Two Modes

| Mode | How | Description |
|------|-----|-------------|
| **Real data** | Add the 5 Kaggle CSVs | Trains on actual F1 history (1950-2023) |
| **Demo mode** | No CSVs needed | Uses synthetic data for demonstration |

## Features

- Select driver, constructor, circuit, grid position, and year
- Instant podium probability prediction
- Grid position sweep chart (probability across all 20 positions)
- Feature importance visualization
- Model configuration panel

## Model

- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Trees**: 100 estimators
- **Max depth**: 10 (pruned to prevent overfitting)
- **Class weight**: Balanced (handles ~12% podium class imbalance)
- **Features**: Grid position, year, front_row flag, laps_ratio, driver, constructor, circuit

## Dataset

Formula 1 World Championship (1950-2023)  
Source: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020  
License: CC0 Public Domain
