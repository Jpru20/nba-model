import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguegamelog

# ==========================================
# 1. CONFIGURATION
# ==========================================
SIMULATIONS = 10000
SEASONS = ['2023-24', '2024-25'] # Use recent data for Pace/Rating

def load_nba_data():
    print("Loading NBA Game Logs for Simulation...")
    games_list = []
    for season in SEASONS:
        try:
            log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='T').get_data_frames()[0]
            log['SEASON_ID'] = season
            games_list.append(log)
            time.sleep(0.6)
        except: pass
        
    return pd.concat(games_list) if games_list else pd.DataFrame()

# ==========================================
# 2. CALCULATE TEAM METRICS (Pace & Efficiency)
# ==========================================
def get_team_metrics(games):
    print("Calculating Pace & Efficiency Metrics...")
    
    # Calculate Possessions (Approx formula)
    # Poss = FGA + 0.44*FTA + TOV - OREB
    games['POSS'] = games['FGA'] + 0.44 * games['FTA'] + games['TOV'] - games['OREB']
    
    # Calculate Offensive Rating (Points per 100 Possessions)
    games['ORTG'] = (games['PTS'] / games['POSS']) * 100
    
    # Group by Team to get Averages and Standard Deviations
    # We use the last 20 games weighted or season average. Here we use full sample mean/std.
    stats = games.groupby('TEAM_ABBREVIATION').agg(
        avg_pace=('POSS', 'mean'),
        std_pace=('POSS', 'std'),
        avg_ortg=('ORTG', 'mean'),
        std_ortg=('ORTG', 'std'),
        games_played=('GAME_ID', 'count')
    ).reset_index()
    
    return stats.set_index('TEAM_ABBREVIATION')

# ==========================================
# 3. SIMULATION LOOP
# ==========================================
def simulate_matchup(home, away, stats, n_sims=5000):
    if home not in stats.index or away not in stats.index: return 0, 0
    
    h = stats.loc[home]
    a = stats.loc[away]
    
    # 1. Estimate Game Pace
    # Average of both teams' paces
    game_pace = (h['avg_pace'] + a['avg_pace']) / 2
    
    # 2. Estimate Efficiency (Points per 100)
    # Home Rating = (Home Off + Away Def) / 2 ... simplifying to Off vs League Avg for now
    # Better: (Home Off Rt + Away Def Rt) - League Avg...
    # Simple Monte Carlo: Use Team's Avg ORTG + HFA
    
    HFA = 3.5 # Home court worth ~3.5 points in NBA
    HFA_RTG = (HFA / game_pace) * 100 # Convert points to Rating boost
    
    h_proj_rtg = h['avg_ortg'] + HFA_RTG
    a_proj_rtg = a['avg_ortg']
    
    # 3. Vectorized Simulation
    # Simulate Pace for this specific game (Normal dist)
    sim_pace = np.random.normal(game_pace, 3.0, n_sims) # 3.0 std dev for pace variance
    
    # Simulate Efficiency for this game
    h_sim_rtg = np.random.normal(h_proj_rtg, h['std_ortg'], n_sims)
    a_sim_rtg = np.random.normal(a_proj_rtg, a['std_ortg'], n_sims)
    
    # Calculate Scores: (Rtg / 100) * Pace
    h_scores = (h_sim_rtg / 100) * sim_pace
    a_scores = (a_sim_rtg / 100) * sim_pace
    
    return np.mean(h_scores - a_scores), np.mean(h_scores + a_scores)

def run_monte_carlo():
    games = load_nba_data()
    if games.empty: return
    stats = get_team_metrics(games)
    
    # Example Matchups (You will automate this in consensus)
    matchups = [
        ('LAL', 'GSW', -2.5, 235.0),
        ('BOS', 'MIA', -8.5, 222.0),
        ('DEN', 'PHX', -4.5, 228.0)
    ]
    
    print("\n" + "="*80)
    print(f"{'MATCHUP':<12} | {'SIM SPREAD':<10} | {'VEGAS':<6} | {'PICK':<10} | {'SIM TOTAL':<10}")
    print("="*80)
    
    for home, away, spread, total in matchups:
        margin, sim_tot = simulate_matchup(home, away, stats, SIMULATIONS)
        
        if margin > 0: p_str = f"{home} by {margin:.1f}"
        else: p_str = f"{away} by {abs(margin):.1f}"
        
        diff = margin - (spread * -1)
        if diff > 0: pick = f"{home} {spread}"
        else: pick = f"{away} {spread}"
        
        print(f"{away} @ {home:<5} | {p_str:<10} | {spread:<6} | {pick:<10} | {sim_tot:.1f}")

if __name__ == "__main__":
    run_monte_carlo()
