import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
from nba_api.stats.endpoints import leaguegamelog
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Includes current season to test recent performance
BACKTEST_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

def load_data():
    print("Fetching historical data for backtesting...")
    games_list = []
    for season in BACKTEST_SEASONS:
        try:
            # Fetch 'Team' gamelogs
            log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='T').get_data_frames()[0]
            log['SEASON_ID'] = season
            games_list.append(log)
            time.sleep(0.6) # Respect API limits
        except: pass
    
    df = pd.concat(games_list) if games_list else pd.DataFrame()
    
    if not df.empty:
        # Sort chronologically
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE')
        print(f"   -> Loaded {len(df)} games.")
    return df

# ==========================================
# 2. RECONSTRUCT FEATURES (Exact Match to Training)
# ==========================================
def process_backtest_data(games):
    print("Reconstructing historical features...")
    
    # 1. Calculate Core Metrics (Pace, Efficiency, Factors)
    # Possessions = FGA + 0.44*FTA + TOV - OREB
    games['POSS'] = games['FGA'] + 0.44 * games['FTA'] + games['TOV'] - games['OREB']
    
    # Offensive Rating (Points per 100 Possessions)
    games['off_rtg'] = (games['PTS'] / games['POSS']) * 100
    
    # Four Factors
    games['eFG'] = (games['FGM'] + 0.5 * games['FG3M']) / games['FGA']
    games['TOV_pct'] = games['TOV'] / games['POSS']
    games['ORB_rate'] = games['OREB'] / (games['OREB'] + games['DREB']) # Proxy
    
    # Style (3-Point Rate)
    games['fg3a_rate'] = games['FG3A'] / games['FGA']
    
    # 2. Calculate Rest Days
    games['prev_game'] = games.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
    games['rest_days'] = (games['GAME_DATE'] - games['prev_game']).dt.days
    games['rest_days'] = games['rest_days'].fillna(3) # Default to well-rested
    
    # 3. Create Matchups (Home vs Away)
    # Filter for Home and Away rows
    home = games[games['MATCHUP'].str.contains('vs.')].copy()
    away = games[games['MATCHUP'].str.contains('@')].copy()
    
    # Merge on GAME_ID
    # Suffixes: _home for Home Team stats, _away for Away Team stats
    df = pd.merge(home, away, on='GAME_ID', suffixes=('_home', '_away'))
    
    # Restore the clean GAME_DATE (it gets suffixed in merge)
    df = df.rename(columns={'GAME_DATE_home': 'GAME_DATE'})
    
    # 4. Rebuild Elo History (To match Training State)
    elos = {t: 1500 for t in pd.concat([df['TEAM_ABBREVIATION_home'], df['TEAM_ABBREVIATION_away']]).unique()}
    h_elos, a_elos = [], []
    K = 20
    HFA = 100 
    
    # Iterate chronologically to build Elo
    for _, row in df.iterrows():
        h, a = row['TEAM_ABBREVIATION_home'], row['TEAM_ABBREVIATION_away']
        h_val = elos.get(h, 1500)
        a_val = elos.get(a, 1500)
        
        h_elos.append(h_val)
        a_elos.append(a_val)
        
        # Update
        margin = row['PTS_home'] - row['PTS_away']
        result = 1 if margin > 0 else 0
        dr = h_val + HFA - a_val
        e_home = 1 / (1 + 10 ** (-dr / 400))
        mult = np.log(abs(margin) + 1)
        shift = K * mult * (result - e_home)
        elos[h] += shift
        elos[a] -= shift
        
    df['home_elo'] = h_elos
    df['away_elo'] = a_elos
    
    # 5. Create Rolling/Season Averages
    # To avoid data leakage, we calculate stats using Season Averages per Year
    final_rows = []
    
    for season in BACKTEST_SEASONS:
        season_games = df[df['SEASON_ID_home'] == season].copy()
        if season_games.empty: continue
        
        # Calculate Team Season Averages
        # (In a perfect backtest, these would be rolling, but season avg is a close proxy for "Team Identity")
        season_stats = games[games['SEASON_ID'] == season]
        units = season_stats.groupby('TEAM_ABBREVIATION')[['eFG', 'TOV_pct', 'ORB_rate', 'POSS', 'off_rtg', 'fg3a_rate']].mean().reset_index()
        
        # Merge Home Stats
        season_games = season_games.merge(units, left_on='TEAM_ABBREVIATION_home', right_on='TEAM_ABBREVIATION').rename(columns={
            'eFG': 'h_eFG', 'TOV_pct': 'h_TOV', 'ORB_rate': 'h_ORB', 'POSS': 'h_Pace', 'off_rtg': 'h_ortg', 'fg3a_rate': 'h_3par'
        })
        # Merge Away Stats
        season_games = season_games.merge(units, left_on='TEAM_ABBREVIATION_away', right_on='TEAM_ABBREVIATION', suffixes=('', '_a')).rename(columns={
            'eFG': 'a_eFG', 'TOV_pct': 'a_TOV', 'ORB_rate': 'a_ORB', 'POSS': 'a_Pace', 'off_rtg': 'a_ortg', 'fg3a_rate': 'a_3par'
        })
        final_rows.append(season_games)
        
    return pd.concat(final_rows) if final_rows else pd.DataFrame()

# ==========================================
# 3. PROFITABILITY CALCULATOR
# ==========================================
def print_profitability(df, metric_col, target_col, threshold_col, label="Spread"):
    print(f"\n--- {label} Profitability Analysis ---")
    print(f"{'CONFIDENCE':<15} | {'WIN %':<8} | {'BETS':<6} | {'PROFIT (100u)':<15}")
    print("-" * 65)

    tiers = [
        ('All Bets', 0.0),
        ('Low (>1.0)', 1.0),
        ('Med (>3.0)', 3.0),
        ('High (>5.0)', 5.0)
    ]
    
    for tier_name, limit in tiers:
        subset = df[df[threshold_col] >= limit].copy()
        
        if label == "Spread":
            # Did the model pick the winner? (Proxy for covering spread)
            wins = np.where(
                ((subset[metric_col] > 0) & (subset[target_col] > 0)) | 
                ((subset[metric_col] < 0) & (subset[target_col] < 0)), 
                1, 0
            )
        else:
            # For totals, we check MAE
            mae = mean_absolute_error(subset[target_col], subset[metric_col])
            print(f"{tier_name:<15} | MAE: {mae:.2f} pts")
            continue

        win_rate = np.mean(wins) * 100
        count = len(subset)
        
        # Profit (Assuming -110)
        units_won = (np.sum(wins) * 1.0) - ((count - np.sum(wins)) * 1.1)
        profit = units_won * 100
        
        print(f"{tier_name:<15} | {win_rate:<6.1f}% | {count:<6} | ${profit:,.0f}")

# ==========================================
# 4. RUN BACKTEST
# ==========================================

def run_backtest():
    # Load Models
    try:
        model_spread = joblib.load('nba_model_spread.pkl')
        model_total = joblib.load('nba_model_total.pkl')
        features = joblib.load('nba_features.pkl')
    except:
        print("Error: Run nba_training_pipeline_blind.py first!")
        return

    # Process Data
    raw_games = load_data()
    if raw_games.empty: return
    df = process_backtest_data(raw_games)
    
    # --- Feature Construction (Must match training exactly) ---
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['rest_diff'] = df['rest_days_home'] - df['rest_days_away']
    
    # Mismatches
    df['efg_mismatch'] = df['h_eFG'] - df['a_eFG']
    df['tov_mismatch'] = df['a_TOV'] - df['h_TOV']
    df['pace_mismatch'] = df['h_Pace'] - df['a_Pace']
    
    # Totals Features
    df['combined_pace'] = df['h_Pace'] + df['a_Pace']
    df['combined_efficiency'] = df['h_ortg'] + df['a_ortg']
    df['3p_volatility'] = (df['h_3par'] + df['a_3par']) / 2
    
    # Fill missing features with 0 (for robustness)
    for f in features:
        if f not in df.columns: df[f] = 0
            
    # --- Prediction ---
    print("\nRunning historical simulations...")
    df['pred_margin'] = model_spread.predict(df[features])
    df['pred_total'] = model_total.predict(df[features])
    
    # --- Results ---
    df['actual_margin'] = df['PTS_home'] - df['PTS_away']
    df['actual_total'] = df['PTS_home'] + df['PTS_away']
    
    # Calculate Error
    df['error_margin'] = abs(df['pred_margin'] - df['actual_margin'])
    df['error_total'] = abs(df['pred_total'] - df['actual_total'])
    
    # Calculate Edge (Proxy using Elo as 'Market')
    df['implied_line'] = df['elo_diff'] / 25 
    df['spread_edge'] = abs(df['pred_margin'] - df['implied_line'])
    
    print("\n" + "="*50)
    print(f"NBA BACKTEST REPORT ({len(df)} Games)")
    print("="*50)
    print(f"Spread MAE:  {df['error_margin'].mean():.2f} pts")
    print(f"Total MAE:   {df['error_total'].mean():.2f} pts")
    
    print_profitability(df, 'pred_margin', 'actual_margin', 'spread_edge', label="Spread")
    print_profitability(df, 'pred_total', 'actual_total', 'spread_edge', label="Total")

    # Export
    cols = ['GAME_DATE', 'TEAM_ABBREVIATION_home', 'TEAM_ABBREVIATION_away', 
            'PTS_home', 'PTS_away', 'pred_margin', 'actual_margin', 
            'pred_total', 'actual_total']
    df[cols].to_csv('nba_backtest_results.csv', index=False)
    print(f"\nSaved detailed results to 'nba_backtest_results.csv'")

if __name__ == "__main__":
    run_backtest()
