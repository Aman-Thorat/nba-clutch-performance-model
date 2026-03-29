# 🏀 NBA Clutch Player Performance Model

A data science project that identifies and predicts "clutch" performance in NBA players using 1M+ shots across 5 seasons (2019–2023).

## Key Findings
- Clutch shots are **3.3% harder** to make (43.2% vs 46.5% FG%)
- **Austin Reaves** ranked #1 clutch performer in 2023 (+23.8% above model expectations)
- **Shot distance** is the strongest predictor of shot success (importance: 0.38)
- Random Forest model achieved **AUC of 0.659** predicting shot outcomes

## Clutch Definition
A shot is clutch if ALL three conditions are met:
- 4th quarter or overtime
- ≤ 5 minutes remaining
- Score margin ≤ 5 points

## Project Structure
```
├── clutch_analysis.ipynb    # Full analysis notebook
├── app.py                   # Streamlit dashboard
└── README.md
```

## Tech Stack
- **Python** — pandas, numpy, scikit-learn
- **Visualization** — matplotlib, seaborn
- **Dashboard** — Streamlit
- **Data** — NBA Play-by-Play 2019–2023 (Kaggle)

## How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit

# Download data from Kaggle:
# https://www.kaggle.com/datasets/[dataset-link]
# Place pbp2019.csv through pbp2023.csv in project folder

# Run the dashboard
streamlit run app.py
```

## Model Performance
| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 60.8% | 0.635 |
| Random Forest | 62.5% | 0.659 |

## Dashboard Features
- 📊 Clutch player rankings with adjustable filters
- 🔍 Individual player deep dive by season
- 📈 Season-level trends across 5 years
EOF
