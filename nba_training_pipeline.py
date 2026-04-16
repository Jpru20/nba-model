import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# --- 1. THE MASK: Monkey-patch standard requests to impersonate Google Chrome ---
import requests
from curl_cffi import requests as curl_requests

class ChromeSession(curl_requests.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, impersonate="chrome")

requests.Session = ChromeSession

# --- 2. NOW import nba_api (it will unknowingly use the Chrome impersonator) ---
from nba_api.stats.endpoints import leaguegamelog, leaguedashplayerstats

# --- 3. Load Proxy Credentials ---
load_dotenv()
PROXY_URL = os.getenv("NBA_PROXY_URL")

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
}

def load_nba_data():
    print("Loading NBA Data via API...")
    games_list = []
    print("    -> Fetching Game Logs...")
    for season in TRAIN_SEASONS:
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season, 
                player_or_team_abbreviation='T',
                headers=custom_headers,
                timeout=30,
                proxy=PROXY_URL
            ).get_data_frames()[0]
            log['SEASON_ID'] = season
            games_list.append(log)
            print(f"        [SUCCESS] {season} Game Logs")
            time.sleep(2.0)
        except Exception as e: 
            print(f"        [ERROR] Failed to fetch Game Logs for {season}: {e}")
    
    games = pd.concat(games_list) if games_list else pd.DataFrame()
    print(f"Success ({len(games)} games).\n")

    print("    -> Fetching Player Stats...")
    player_list = []
    for season in TRAIN_SEASONS:
        try:
            p_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                headers=custom_headers,
                timeout=30,
                proxy=PROXY_URL
            ).get_data_frames()[0]
            p_stats['SEASON_ID'] = season
            player_list.append(p_stats)
            print(f"        [SUCCESS] {season} Player Stats")
            time.sleep(2.0)
        except Exception as e: 
            print(f"        [ERROR] Failed to fetch Player Stats for {season}: {e}")
    
    players = pd.concat(player_list) if player_list else pd.DataFrame()
    print("Success.")
    return games, players

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def calculate_advanced_stats(df):
    print("Calculating Four Factors & Advanced Metrics...")
    df = df.copy()
    
    # 1. Pace (Possessions)
    df['POSS'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV'] - df['OREB']
    
    # 2. Efficiency (Off Rating)
    df['off_rtg'] = (df['PTS'] / df['POSS']) * 100
    
    # 3. Four Factors
    df['eFG'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['TOV_pct'] = df['TOV'] / df['POSS']
    df['ORB_rate'] = df['OREB'] / (df['OREB'] + df['DREB'])
    df['FT_rate'] = df['FTA'] / df['FGA']
    df['fg3a_rate'] = df['FG3A'] / df['FGA']
    
    return df

def calculate_elo_and_rest(games):
    print("Calculating Elo & Rest Days...")
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE')
    
    games['prev_game'] = games.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
    games['rest_days'] = (games['GAME_DATE'] - games['prev_game']).dt.days
    games['rest_days'] = games['rest_days'].fillna(3)
    
    home_games = games[games['MATCHUP'].str.contains('vs.')].copy()
    away_games = games[games['MATCHUP'].str.contains('@')].copy()
    df = pd.merge(home_games, away_games, on='GAME_ID', suffixes=('_home', '_away'))
    
    elos = {t: 1500 for t in pd.concat([df['TEAM_ABBREVIATION_home'], df['TEAM_ABBREVIATION_away']]).unique()}
    h_elos, a_elos = [], []
    K = 20
    HFA = 100 
    
    for _, row in df.iterrows():
        h = row['TEAM_ABBREVIATION_home']
        a = row['TEAM_ABBREVIATION_away']
        h_val = elos.get(h, 1500); a_val = elos.get(a, 1500)
        h_elos.append(h_val); a_elos.append(a_val)
        
        margin = row['PTS_home'] - row['PTS_away']
        result = 1 if margin > 0 else 0
        dr = h_val + HFA - a_val
        e_home = 1 / (1 + 10 ** (-dr / 400))
        mult = np.log(abs(margin) + 1)
        shift = K * mult * (result - e_home)
        elos[h] += shift; elos[a] -= shift
        
    df['home_elo'] = h_elos
    df['away_elo'] = a_elos
    joblib.dump(elos, 'nba_elo_state.pkl')
    return df

def calculate_player_values(players_df):
    print("Calculating Player Value...")
    if players_df.empty: return pd.DataFrame()
    if 'PIE' in players_df.columns:
        players_df['value_per_game'] = players_df['PIE'] * 100
    else:
        players_df['value_per_game'] = (players_df['PTS'] + players_df['REB'] + players_df['AST'] - players_df['TOV']) / players_df['GP']
    latest = players_df['SEASON_ID'].max()
    current = players_df[players_df['SEASON_ID'] == latest]
    current[['PLAYER_NAME', 'value_per_game']].to_csv("nba_player_values.csv", index=False)
    return current

def run_pipeline():
    games, players = load_nba_data()
    if games.empty: return

    games = calculate_advanced_stats(games)
    
    latest_season = games['SEASON_ID'].max()
    recent = games[games['SEASON_ID'] == latest_season].copy()
    recent['GAME_DATE'] = pd.to_datetime(recent['GAME_DATE'])
    recent = recent.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # --- DUAL-FEATURE IMPLEMENTATION START ---
    target_cols = ['eFG', 'TOV_pct', 'ORB_rate', 'FT_rate', 'POSS', 'off_rtg', 'fg3a_rate']
    
    # 1. Season Average (Class/Baseline)
    unit_stats_avg = recent.groupby('TEAM_ABBREVIATION')[target_cols].mean().reset_index()
    
    # 2. 14-game EMA (Momentum/Form)
    unit_stats_ema = recent.groupby('TEAM_ABBREVIATION')[target_cols].apply(
        lambda x: x.ewm(span=14, min_periods=1).mean().tail(1)
    ).reset_index(level=0)
    unit_stats_ema.columns = ['TEAM_ABBREVIATION'] + ['ema_' + c for c in target_cols]
    
    # Combine them
    unit_stats = pd.merge(unit_stats_avg, unit_stats_ema, on='TEAM_ABBREVIATION')
    # --- DUAL-FEATURE IMPLEMENTATION END ---
    
    unit_stats.to_csv("nba_unit_stats.csv", index=False)
    
    calculate_player_values(players)
    df = calculate_elo_and_rest(games)
    
    # Merge Unit Stats for Home and Away
    df = df.merge(unit_stats, left_on='TEAM_ABBREVIATION_home', right_on='TEAM_ABBREVIATION').rename(columns={
        'eFG': 'h_eFG', 'TOV_pct': 'h_TOV', 'ORB_rate': 'h_ORB', 'POSS': 'h_Pace', 'off_rtg': 'h_ortg', 'fg3a_rate': 'h_3par',
        'ema_eFG': 'h_ema_eFG', 'ema_TOV_pct': 'h_ema_TOV', 'ema_ORB_rate': 'h_ema_ORB', 'ema_POSS': 'h_ema_Pace', 'ema_off_rtg': 'h_ema_ortg'
    })
    df = df.merge(unit_stats, left_on='TEAM_ABBREVIATION_away', right_on='TEAM_ABBREVIATION', suffixes=('', '_a')).rename(columns={
        'eFG': 'a_eFG', 'TOV_pct': 'a_TOV', 'ORB_rate': 'a_ORB', 'POSS': 'a_Pace', 'off_rtg': 'a_ortg', 'fg3a_rate': 'a_3par',
        'ema_eFG': 'a_ema_eFG', 'ema_TOV_pct': 'a_ema_TOV', 'ema_ORB_rate': 'a_ema_ORB', 'ema_POSS': 'a_ema_Pace', 'ema_off_rtg': 'a_ema_ortg'
    })
    
    # Create Features
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['rest_diff'] = df['rest_days_home'] - df['rest_days_away']
    
    # Season Baseline Diffs
    df['efg_mismatch'] = df['h_eFG'] - df['a_eFG']
    df['tov_mismatch'] = df['a_TOV'] - df['h_TOV']
    df['pace_mismatch'] = df['h_Pace'] - df['a_Pace']
    
    # Momentum (EMA) Diffs
    df['ema_efg_mismatch'] = df['h_ema_eFG'] - df['a_ema_eFG']
    df['ema_tov_mismatch'] = df['a_ema_TOV'] - df['h_ema_TOV']
    
    df['combined_pace'] = df['h_Pace'] + df['a_Pace']
    df['combined_efficiency'] = df['h_ortg'] + df['a_ortg']
    df['3p_volatility'] = (df['h_3par'] + df['a_3par']) / 2
    
    df['target_spread'] = df['PTS_home'] - df['PTS_away']
    df['target_total'] = df['PTS_home'] + df['PTS_away']
    
    train = df.dropna(subset=['target_spread'])
    
    print(f"Training NBA Model on {len(train)} games...")
    features = [
        'elo_diff', 'home_elo', 'away_elo', 
        'rest_diff', 'rest_days_home', 'rest_days_away',
        'efg_mismatch', 'tov_mismatch', 'pace_mismatch',
        'ema_efg_mismatch', 'ema_tov_mismatch', # New Dual-Features
        'combined_pace', 'combined_efficiency', '3p_volatility'
    ]
    
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target_spread'], test_size=0.15)
    
    # Train Spread
    model = xgb.XGBRegressor(n_estimators=1500, max_depth=5, learning_rate=0.01)
    model.fit(X_train, y_train)
    joblib.dump(model, 'nba_model_spread.pkl')
    
    # Train Total
    model_tot = xgb.XGBRegressor(n_estimators=1500, max_depth=5, learning_rate=0.01)
    model_tot.fit(train[features], train['target_total'])
    joblib.dump(model_tot, 'nba_model_total.pkl')
    
    joblib.dump(features, 'nba_features.pkl')
    
    # --- PRINT BRAIN SCAN ---
    print("\n" + "="*40)
    print("    MODEL BRAIN PRIORITY      ")
    print("="*40)
    
    for name, m in [("Spread Model", model), ("Total Model", model_tot)]:
        print(f"\n--- {name} ---")
        imp = m.get_booster().get_score(importance_type='gain')
        total = sum(imp.values())
        for k, v in sorted(imp.items(), key=lambda x:x[1], reverse=True):
            print(f"{k:<20}: {(v/total)*100:.1f}%")

    print("\nTraining Complete.")

if __name__ == "__main__":
    run_pipeline()
