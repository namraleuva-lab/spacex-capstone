# SpaceX Falcon 9 First Stage Landing Prediction

IBM Applied Data Science Capstone. Predicts whether the Falcon 9 first stage lands successfully.

## Contents
- `analysis.py` : EDA charts, SQL queries (SQLite), GridSearchCV models (Logistic Regression, SVM, Decision Tree, KNN), dashboard charts
- `spacex_dash_app.py` : Plotly Dash dashboard (site dropdown, success pie chart, payload slider, scatter)
- `data/` : dataset_part_2.csv, dataset_part_3.csv, spacex_launch_dash.csv
- `charts/` : generated figures
- `results.json` : numeric results (accuracies, SQL outputs, success rates)
- `report/` : final capstone presentation (PDF)

## Run
```
pip install -r requirements.txt
python analysis.py
python spacex_dash_app.py
```

## Key results
- 90 Falcon 9 launches, overall landing success rate 66.7%
- Success rate rose from 0% (2010-2013) to about 90% (2019-2020)
- KSC LC-39A has the best site record (77%)
- Logistic Regression, SVM and KNN reach 83.3% test accuracy; SVM chosen on cross-validation (84.8%)
