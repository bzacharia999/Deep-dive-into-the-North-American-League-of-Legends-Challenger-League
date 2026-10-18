# Study of NA's Top League of Legends Players

This repository serves as a case study analyzing the top Challenger players in North America's League of Legends ladder. This project ingests data directly from the Riot Games API, performs extensive exploratory data analysis on player and match-level statistics, and trains predictive models to forecast match outcomes (`win`).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Workflow & Methodology](#workflow--methodology)
  - [1. Data Collection (`01_data_collection.ipynb`)](#1-data-collection)
  - [2. Exploratory Data Analysis (`02_eda.ipynb`)](#2-exploratory-data-analysis)
  - [3. Predictive Modelling (`03_modelling.ipynb`)](#3-predictive-modelling)
- [Key Insights & Findings](#key-insights--findings)
- [Model Evaluation & Results](#model-evaluation--results)
- [Installation & Setup](#installation--setup)
- [Technologies & Libraries](#technologies--libraries)

---

## Project Overview

In competitive League of Legends, match outcomes depend on a complex interplay between in-game performance metrics (e.g., gold generation, objective control, champion combat efficiency) and player characteristics (e.g., season win rate, total games played, ladder rank).

The primary goals of this project are:
1. **Automated Pipeline**: Build a resilient, rate-limited data extraction pipeline using Riot Games' `league-v4` and `match-v5` APIs.
2. **Behavioral & Meta Analysis**: Investigate how champion classes (Fighter, Mage, Marksman, Assassin, Tank, Support) perform across varying game durations, whether time of day impacts player win rate, etc.
3. **Outcome Prediction**: Build and evaluate classification models ([Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and [XGBoost](https://xgboost.readthedocs.io/)) to predict match wins and analyze what factors are most important to winning/losing.

---

## Repository Structure

```text
Study-of-NA-s-top-100-LoL-players/
├── data/
│   ├── raw/                       # Cached Riot API & Data Dragon responses
│   └── processed/
│       ├── top300_players.csv     # Ladder-level player dataset (300 players)
│       └── player_matches.csv     # Match-level participant dataset (30,000 observations)
├── notebooks/
│   ├── 01_data_collection.ipynb   # API ingestion, rate limiting, and raw-to-processed ETL
│   ├── 02_eda.ipynb               # Exploratory data analysis into the different factors of a match
│   └── 03_modelling.ipynb         # Feature engineering, CV tuning, modeling, and evaluation
├── src/
│   ├── config.py                  # API endpoints, queue routing, rate limits, and directory paths
│   └── riot_api.py                # Rate-limited HTTP client with backoff and disk caching
├── .env.example                   # Template for Riot API key configuration
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## Workflow & Methodology

### 1. Data Collection
Implemented in [01_data_collection.ipynb](file:///c:/Users/Ben/Documents/Data%20Science%20Projects/Study-of-NA-s-top-100-LoL-players/notebooks/01_data_collection.ipynb) and supported by [src/riot_api.py](file:///c:/Users/Ben/Documents/Data%20Science%20Projects/Study-of-NA-s-top-100-LoL-players/src/riot_api.py):

- **Riot Games API Integration**:
  - `league-v4`: Extracted the top 300 NA Challenger players in Ranked Solo/Duo (`RANKED_SOLO_5x5`).
  - `match-v5`: Extracted the 100 most recent match IDs per player via PUUID routing across the Americas region.
  - `match-v5 /matches/{matchId}`: Retrieved detailed telemetry and participant metrics for each match.
- **Resilience & Rate Limiting**:
  - Implemented a two-tier sliding window rate limiter enforcing Riot's developer limits (20 requests/second and 100 requests/120 seconds).
- **Processed Datasets Exported**:
  - `top300_players.csv`: Ladder rank, League Points (LP), wins, losses, win rate, hot streak, and Riot ID.
  - `player_matches.csv`: 30,000 player-match observations containing kills, deaths, assists, damage dealt to champions and objectives, vision scores, gold per minute, CS per minute, turret kills, and remake flags.

---

### 2. Exploratory Data Analysis
Implemented in [02_eda.ipynb](file:///c:/Users/Ben/Documents/Data%20Science%20Projects/Study-of-NA-s-top-100-LoL-players/notebooks/02_eda.ipynb):

- **Data Dragon Champion Tagging**:
  - Dynamically fetched champion metadata from Riot's Data Dragon endpoint to assign primary classes (`Fighter`, `Mage`, `Marksman`, `Assassin`, `Tank`, `Support`).
- **Remake & Missing Value Handling**:
  - Identified and removed 281 early remake games.
- **Core Hypotheses Examined**:
  - *Game Duration Bimodality*: Game length exhibits a bimodal distribution with a sharp spike at 15 minutes (earliest early surrender threshold) and a main peak around 27 minutes.
  - *Class Win Rates vs. Game Duration*: Different classes achieve different win rates depending on game length, as classes often vary in strength based on the amount of gold/ champion level
  - *K/D Ratios by Class*: Damage-oriented roles lead K/D ratios.
  - *Ladder LP Driver*: Win rate correlates significantly stronger with higher League Points than sheer volume of games played.
  - *Time-of-Day Effect*: Hourly win rates and average deaths remain consistent across all 24 hours of the day in the challenger league
  - *Linear Correlations with Win*: Correlation matrix analysis highlighted `kd_ratio`, `damageDealtToObjectives`, `turretKills`, and `gold_per_min` as the strongest linear predictors of match victory.

---

### 3. Predictive Modelling
Implemented in [03_modelling.ipynb](file:///c:/Users/Ben/Documents/Data%20Science%20Projects/Study-of-NA-s-top-100-LoL-players/notebooks/03_modelling.ipynb):

- **Target vs. Feature Visualizations**:
  - Analyzed feature densities comparing wins and losses via translucent histograms overlaid with Kernel Density Estimation (KDE) curves.
  - Evaluated categorical win/loss proportions for `primary_class` and `firstBloodKill`.
- **Validation Strategy & Evaluation Metrics**:
  - **5-Fold Cross-Validation** applied across all models.
  - Hyperparameter tuning and model selection determined by **ROC-AUC** on out-of-fold probability estimates.
  - Metrics tracked across all folds: **ROC-AUC**, **Accuracy** (threshold 0.5), **Precision**, and **Recall**.
- **Model Interpretability**:
  - Extracted standardized log-odds coefficients for Logistic Regression.
  - Generated side-by-side feature importance bar charts (Gini Impurity for Random Forest, Gain for XGBoost).

---

## Key Insights & Findings

| Observation | Finding |
|---|---|
| **Primary Win Drivers** | In-game objective metrics (`turretKills`, `damageDealtToObjectives`) and economic acceleration (`gold_per_min`, `kd_ratio`) dominate all models' feature importances. |
| **Player vs. Match Stats** | Match-specific performance metrics overwhelmingly outweigh season-long player ladder stats (`win_rate`, `leaguePoints`, `games`) when predicting individual game outcomes. |
| **Model Superiority** | Non-linear tree ensembles (Random Forest and XGBoost) outperform linear logistic regression by capturing non-linear threshold effects (e.g., objective damage spikes and gold leads). |

---

## Model Evaluation & Results

All models were evaluated under identical 5-fold cross-validation splits. Summary of out-of-fold performance:

| Model Architecture | Hyperparameters | 5-Fold ROC-AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | $\lambda^* \approx 0.0464$ ($C \approx 21.5$) | ~0.90 – 0.91 | ~82 – 83% | ~82 – 84% | ~81 – 83% |
| **Random Forest** | `max_depth` = 10 – 14 | ~0.92 – 0.93 | ~84 – 85% | ~84 – 86% | ~83 – 85% |
| **XGBoost** | `max_depth` = 5 – 7 | **~0.93 – 0.94** | **~85 – 86%** | **~85 – 86%** | **~84 – 86%** |

*Note: Exact figures vary marginally depending on random seed and fold partitioning. XGBoost consistently delivers the highest ROC-AUC.*


## Technologies & Libraries

- **API & Networking**: `requests`, `python-dotenv`
- **Data Manipulation & Storage**: `pandas`, `numpy`, `pyarrow`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning & Modeling**: `scikit-learn`, `xgboost`
- **Progress Tracking**: `tqdm`