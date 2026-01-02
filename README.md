# NBA Consensus Model & Scraper 🏀

A machine learning-based NBA betting predictor that combines **XGBoost**, **Elo Ratings**, and **Real-Time Injury Analysis** to forecast spreads and totals.

This project features a robust consensus engine that adjusts for the "Modern NBA" scoring environment (2025-26 Season) and scrapes official NBA PDF injury reports to calculate precise player impact penalties.

## 🚀 Key Features

* **Dual-Model Consensus:** Blends XGBoost predictions with a Monte Carlo-style simulation based on Pace and Offensive Rating.
* **Smart Injury Scraper:** Automatically downloads and parses official NBA PDF injury reports, handling condensed formatting and matching team names.
* **Dynamic Injury Impact:** Calculates a "Loss Value" for every team based on the PIE (Player Impact Estimate) of missing players.
    * *Example:* If a Superstar (PIE > 0.18) is OUT, the model penalizes the spread by ~5.5 points.
* **Modern Scoring Patch:** Adjusted for 2025 efficiency (1.16x Pace multiplier) and High-Volume 3-Point shooting bonuses.
* **Live Odds Integration:** Fetches real-time lines from The Odds API to identify "High Confidence" value edges.

## 📂 File Structure

* `nba_pure_consensus.py` - **The Main Predictor.** Runs the model, fetches odds, and prints the final betting card.
* `scrape_nba_injuries.py` - **The Scraper.** Downloads the latest PDF from NBA.com and updates `data/nba_injuries.json`.
* `nba_training_pipeline.py` - Script used to retrain the XGBoost models on new data.
* `nba_model_spread.pkl` / `nba_model_total.pkl` - The trained machine learning models.
* `nba_player_values.csv` - Database of player PIE ratings used for injury calculations.
* `nba_unit_stats.csv` - Team-level advanced stats (eFG%, TOV%, Pace).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jpru20/nba-model.git](https://github.com/Jpru20/nba-model.git)
    cd nba-model
    ```

2.  **Install Dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy requests joblib scikit-learn xgboost nba_api pytz pymupdf
    ```

3.  **Set Up API Keys:**
    * Open `nba_pure_consensus.py`.
    * Replace `ODDS_API_KEY` with your free key from [The Odds API](https://the-odds-api.com/).

## ⚡ Usage

**Step 1: Update Injuries**
Run the scraper to get the latest report from the NBA.
```bash
python scrape_nba_injuries.py

Step 2: Generate Picks Run the consensus model to see today's predictions.

Bash

python nba_pure_consensus.py

Sample Output

MATCHUP      | SCORE    | XGB   | CON   | VEGAS  | PICK        | EDGE  | CONF  | ANALYSIS
--------------------------------------------------------------------------------------------------
LAL @ BOS    | 112-118  | -6.5  | -5.8  | -5.5   | BOS -5.5    | 0.3   | LOW   | BOS Elo Adv; LAL B2B
GSW @ PHX    | 107-107  | +2.1  | +0.5  | -1.5   | GSW +1.5    | 2.0   | MED   | Inj Impact (PHX-5.1)

