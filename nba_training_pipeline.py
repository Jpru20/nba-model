import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
from nba_api.stats.endpoints import leaguegamelog, leaguedashplayerstats
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATED: Includes current 2025-26 season
TRAIN_SEASONS = ['2023-24', '2024-25', '2025-26']

def load_nba_data():
    print(f"Loading NBA Data ({TRAIN_SEASONS})...")
    
    # 1. Game Logs
    print("   -> Fetching Game Logs...", end=" ")
    games_list = []
    for season in TRAIN_SEASONS:
        try:
            # 'T' = Team Stats
            log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='T').get_data_frames()[0]
            log['SEASON_ID'] = season
            games_list.append(log)
            time.sleep(0.6)
        except: pass
    
    games = pd.concat(games_list) if games_list else pd.DataFrame()
    print(f"Success ({len(games)} games).")

    # 2. Player Stats
    print("   -> Fetching Player Stats...", end=" ")
    player_list = []
    for season in TRAIN_SEASONS:
        try:
            p_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
            p_stats['SEASON_ID'] = season
            player_list.append(p_stats)
            time.sleep(0.6)
        except: pass
    
    players = pd.concat(player_list) if player_list else pd.DataFrame()
    print("Success.")
    
    return games, players

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def calculate_advanced_stats(df):
    print("Calculating Four Factors & Advanced Metrics...")
    df = df.copy()
    
    # Pace Estimate (Possessions)
    # Formula: FGA + 0.44*FTA + TOV - OREB
    df['POSS'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV'] - df['OREB']
    
    # Efficiency (Offensive Rating)
    df['off_rtg'] = (df['PTS'] / df['POSS']) * 100
    
    # Four Factors
    df['eFG'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['TOV_pct'] = df['TOV'] / df['POSS']
    df['ORB_rate'] = df['OREB'] / (df['OREB'] + df['DREB']) # Proxy
    df['FT_rate'] = df['FTA'] / df['FGA']
    
    # Style (3-Point Rate)
    df['fg3a_rate'] = df['FG3A'] / df['FGA']
    
    return df

def calculate_elo_and_rest(games):
    print("Calculating Elo & Rest Days...")
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE')
    
    # Rest Days
    games['prev_game'] = games.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
    games['rest_days'] = (games['GAME_DATE'] - games['prev_game']).dt.days
    games['rest_days'] = games['rest_days'].fillna(3)
    
    # Matchups
    home_games = games[games['MATCHUP'].str.contains('vs.')].copy()
    away_games = games[games['MATCHUP'].str.contains('@')].copy()
    df = pd.merge(home_games, away_games, on='GAME_ID', suffixes=('_home', '_away'))
    
    # Elo
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
    
    # PIE (Player Impact Estimate)
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

    # 1. Feature Engineering
    games = calculate_advanced_stats(games)
    
    # Save Unit Stats (Most recent season averages)
    latest_season = games['SEASON_ID'].max()
    recent = games[games['SEASON_ID'] == latest_season]
    
    # Save POSS and off_rtg for Consensus
    unit_stats = recent.groupby('TEAM_ABBREVIATION')[['eFG', 'TOV_pct', 'ORB_rate', 'FT_rate', 'POSS', 'off_rtg', 'fg3a_rate']].mean().reset_index()
    unit_stats.to_csv("nba_unit_stats.csv", index=False)
    
    calculate_player_values(players)
    
    # 2. Merge & Train
    df = calculate_elo_and_rest(games)
    
    # Merge Unit Stats
    df = df.merge(unit_stats, left_on='TEAM_ABBREVIATION_home', right_on='TEAM_ABBREVIATION').rename(columns={
        'eFG': 'h_eFG', 'TOV_pct': 'h_TOV', 'ORB_rate': 'h_ORB', 'POSS': 'h_Pace', 'off_rtg': 'h_ortg', 'fg3a_rate': 'h_3par'
    })
    df = df.merge(unit_stats, left_on='TEAM_ABBREVIATION_away', right_on='TEAM_ABBREVIATION', suffixes=('', '_a')).rename(columns={
        'eFG': 'a_eFG', 'TOV_pct': 'a_TOV', 'ORB_rate': 'a_ORB', 'POSS': 'a_Pace', 'off_rtg': 'a_ortg', 'fg3a_rate': 'a_3par'
    })
    
    # 3. Create Features
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['rest_diff'] = df['rest_days_home'] - df['rest_days_away']
    
    df['efg_mismatch'] = df['h_eFG'] - df['a_eFG']
    df['tov_mismatch'] = df['a_TOV'] - df['h_TOV']
    df['pace_mismatch'] = df['h_Pace'] - df['a_Pace']
    
    # Totals Features
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
        'combined_pace', 'combined_efficiency', '3p_volatility'
    ]
    
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target_spread'], test_size=0.15)
    
    # Train Spread
    model = xgb.XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01)
    model.fit(X_train, y_train)
    joblib.dump(model, 'nba_model_spread.pkl')
    
    # Train Total
    model_tot = xgb.XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01)
    model_tot.fit(train[features], train['target_total'])
    joblib.dump(model_tot, 'nba_model_total.pkl')
    
    joblib.dump(features, 'nba_features.pkl')
    print("NBA Training Complete.")

if __name__ == "__main__":
    run_pipeline()
