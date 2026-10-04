# Weather Rain Prediction

A beginner-friendly machine learning project that predicts whether it rains in a given hour from weather readings. It compares **KNN**, **Logistic Regression** and **Naive Bayes**, and shows why **accuracy alone is misleading** on imbalanced data.

> **Note:** The CSV files in `data/` are **synthetic practice data** generated to mimic a dry January and a monsoon-like August. They are not real measurements, so do not draw real weather conclusions from the results.

## Goal

Predict `rained` (1 = rain, 0 = no rain) using:

| Feature | Description |
|---|---|
| `temperature` | Air temperature (°C) |
| `humidity` | Relative humidity (%) |
| `wind_speed` | Wind speed |
| `pressure` | Air pressure (hPa) |

`rain_mm` is **not** used as a feature, because it directly reveals the answer (data leakage).

## Dataset

| File | Rows | Description |
|---|---|---|
| `data/weather_jan.csv` | 240 | Hourly data, 1-10 Jan 2025 (dry) |
| `data/weather_aug.csv` | 241 | Hourly data, 1-10 Aug 2025 (wet), includes 1 duplicate row |

Columns: `timestamp, temperature, humidity, wind_speed, pressure, rain_mm, rained`

After combining and removing the duplicate: **480 rows**, with about **77.5% no rain** and **22.5% rain** (imbalanced).

## Workflow

1. **Combine** both files with `pd.concat(..., ignore_index=True)`
2. **Clean:** remove duplicates, fill missing values with the median
3. **Check the target:** `value_counts()` shows the class imbalance
4. **Split:** 80/20 with `stratify=y` so both sets keep the same rain share
5. **Scale:** `StandardScaler` (fit on train only) for KNN and Logistic Regression
6. **Train** three models and compare them against a baseline
7. **Evaluate** with confusion matrix, precision, recall and F1 (not accuracy alone)

## Models

| Model | Scaling | Notes |
|---|---|---|
| Baseline (always "no rain") | No | About 77% accuracy, catches no rain |
| Naive Bayes (`GaussianNB`) | No | Fast probabilistic baseline |
| KNN | Yes | Distance-based, sensitive to `k` |
| Logistic Regression | Yes | Tested with `class_weight="balanced"` |

## Results

Single 80/20 split, 96 test rows (74 dry, 22 rainy). Rain = class 1.

| Model | Accuracy | Rain precision | Rain recall | Rain F1 |
|---|---|---|---|---|
| KNN | 0.72 | 0.31 | 0.18 | 0.23 |
| Logistic Regression | 0.61 | 0.36 | 0.91 | 0.52 |
| Naive Bayes | _add your result_ | | | |

### Key takeaways

- A model that always predicts "no rain" already scores about 77%, so **accuracy alone is misleading**.
- KNN had higher accuracy but caught only **4 of 22** rainy hours.
- Logistic Regression had lower accuracy but caught **20 of 22** rainy hours, at the cost of more false alarms (35).
- With only 22 rainy test rows, results are noisy. Cross-validation gives a more reliable comparison.

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/arifusman7/weather-rain-prediction.git
cd weather-rain-prediction

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/rain_prediction.ipynb
```

You can also open the notebook in **Google Colab** and upload the two CSV files from `data/`.

## Project structure

```
weather-rain-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── weather_jan.csv
│   └── weather_aug.csv
└── notebooks/
    └── rain_prediction.ipynb
```

## What I learned

- Combining files with `concat` and why `ignore_index=True` matters
- When encoding is needed (not for this all-numeric data) and which models need scaling
- Why `stratify=y` helps on imbalanced classification
- How to read a confusion matrix and judge a model by recall and F1 on the rare class
- How `class_weight="balanced"` trades false alarms for catching more rain

## Possible improvements

- Add `hour` (or `sin`/`cos` of the hour) as a feature
- Use cross-validation and `GridSearchCV` to tune `k` and `C`
- Try Random Forest and Gradient Boosting
- Report ROC-AUC and tune the prediction threshold
- Replace the synthetic data with real station data

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter

## License

This project is for learning purposes. Add a license (for example MIT) if you want others to reuse it.
