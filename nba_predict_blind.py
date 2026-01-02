import joblib
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

# ==========================================
# 1. CONFIGURATION
# ==========================================
ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae" # Your Key

# ==========================================
# 2. LIVE DATA FETCHING
# ==========================================
def get_live_odds():
    print("   [Odds API] Fetching NBA lines...")
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'api_key': ODDS_API_KEY,
        'regions': 'us', 'markets': 'spreads,totals', 'oddsFormat': 'american', 'bookmakers': 'draftkings'
    }
    try:
        res = requests.get(url, params=params).json()
        matchups = []
        
        # Map Names to Abbr
        team_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        
        for game in res:
            h = team_map.get(game['home_team'])
            a = team_map.get(game['away_team'])
            if not h or not a: continue
            
            spread = 0.0
            total = 225.0
            
            # Parse Odds
            if game['bookmakers']:
                markets = game['bookmakers'][0]['markets']
                for m in markets:
                    if m['key'] == 'spreads':
                        for o in m['outcomes']:
                            if o['name'] == game['home_team']: spread = o['point']
                    if m['key'] == 'totals':
                        total = m['outcomes'][0]['point']
            
            matchups.append({'home': h, 'away': a, 'spread': float(spread), 'total': float(total)})
        return matchups
    except:
        print("   [Error] Odds API failed.")
        return []

def scrape_injuries():
    print("   [Scraper] Checking CBS NBA Injury Report...")
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        injuries = {}
        
        def get_abbr(name):
            if "Brooklyn" in name: return "BKN"
            if "Charlotte" in name: return "CHA"
            if "Lakers" in name: return "LAL"
            if "Clippers" in name: return "LAC"
            if "Pelicans" in name: return "NOP"
            if "Spurs" in name: return "SAS"
            if "Knicks" in name: return "NYK"
            if "Thunder" in name: return "OKC"
            if "Warriors" in name: return "GSW"
            return name[:3].upper()

        for link in soup.find_all('a', href=True):
            if '/nba/teams/' in link['href']:
                team = get_abbr(link.text.strip())
                if team not in injuries: injuries[team] = {}
                
                table = link.find_next('table')
                if table:
                    for row in table.find_all('tr'):
                        cols = row.find_all('td')
                        if len(cols) > 4:
                            name = cols[0].text.strip()
                            status = cols[4].text.lower()
                            
                            code = None
                            if 'out' in status: code = 'OUT'
                            elif 'doubtful' in status: code = 'DOUBTFUL'
                            elif 'questionable' in status: code = 'QUESTIONABLE'
                            elif 'day-to-day' in status: code = 'GTD'
                            
                            if code: injuries[team][name] = code
        return injuries
    except: return {}

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================

def explain_pick(row, home, away, impact, pred_total, vegas_total):
    reasons = []
    
    # 1. Elo
    elo_diff = row['elo_diff'].values[0]
    if elo_diff > 50: reasons.append(f"{home} Elo Edge")
    elif elo_diff < -50: reasons.append(f"{away} Elo Edge")
    
    # 2. Pace / Style
    pace_mis = row['pace_mismatch'].values[0]
    vol = row['3p_volatility'].values[0]
    
    if vol > 0.45: reasons.append("High Variance (3-Point Heavy)")
    
    # 3. Totals Analysis
    total_diff = pred_total - vegas_total
    if total_diff > 4: 
        reasons.append(f"Fast Pace / High Efficiency (Ov {pred_total:.0f})")
    elif total_diff < -4: 
        reasons.append(f"Slow Pace / Defense (Un {pred_total:.0f})")
    
    # 4. Injuries
    if abs(impact) > 1.5:
        team = home if impact < 0 else away
        reasons.append(f"Load Management: {team} hurt (-{abs(impact):.1f})")
        
    return "; ".join(reasons) if reasons else "Stats Edge"

def run_predictions():
    print("\nLoading NBA Brain...")
    try:
        model = joblib.load('nba_model_spread.pkl')
        model_tot = joblib.load('nba_model_total.pkl')
        elo = joblib.load('nba_elo_state.pkl')
        unit_stats = pd.read_csv('nba_unit_stats.csv')
        player_vals = pd.read_csv('nba_player_values.csv')
        features = joblib.load('nba_features.pkl')
    except:
        print("Error: Run nba_training_pipeline_blind.py first.")
        return

    matchups = get_live_odds()
    injuries = scrape_injuries()
    
    # Helper: Injury Impact
    def get_impact(team, team_injuries):
        if not team_injuries: return 0.0
        loss = 0.0
        weights = {'OUT': 1.0, 'DOUBTFUL': 0.75, 'QUESTIONABLE': 0.5, 'GTD': 0.25}
        
        for p, status in team_injuries.items():
            last = p.split(' ')[-1]
            matches = player_vals[player_vals['PLAYER_NAME'].str.contains(last, case=False)]
            w = weights.get(status, 0.5)
            
            if not matches.empty:
                val = matches['value_per_game'].max()
                loss += (val * w * 0.15) # Approx 0.15 pts per value unit
            else:
                loss += (5.0 * w * 0.15) # Bench penalty
        return loss

    print("\n" + "="*145)
    print(f"{'MATCHUP':<12} | {'TRUE LINE':<10} | {'VEGAS':<6} | {'PICK':<10} | {'EDGE':<6} | {'TRUE TOT':<8} | {'VEGAS':<6} | {'PICK':<10} | {'EDGE':<6} | {'ANALYSIS'}")
    print("="*145)

    for m in matchups:
        home, away = m['home'], m['away']
        spread = m['spread']
        vegas_total = m['total']
        
        # Stats
        h_elo = elo.get(home, 1500)
        a_elo = elo.get(away, 1500)
        
        try:
            h_u = unit_stats[unit_stats['TEAM_ABBREVIATION'] == home].iloc[0]
            a_u = unit_stats[unit_stats['TEAM_ABBREVIATION'] == away].iloc[0]
            
            # Mismatches
            pace_m = h_u['possessions'] - a_u['possessions']
            eff_m = h_u['off_rtg'] - a_u['off_rtg']
            vol_3p = (h_u['fg3a_rate'] + a_u['fg3a_rate']) / 2
            
            efg_d = h_u['eFG'] - a_u['eFG']
            tov_d = a_u['tov_pct'] - h_u['tov_pct']
        except:
            pace_m, eff_m, vol_3p, efg_d, tov_d = 0, 0, 0, 0, 0
        
        # Injuries
        h_loss = get_impact(home, injuries.get(home, {}))
        a_loss = get_impact(away, injuries.get(away, {}))
        net_impact = a_loss - h_loss # Positive = Home Advantage

        # Predict
        row = pd.DataFrame([{
            'elo_diff': h_elo - a_elo, 'home_elo': h_elo, 'away_elo': a_elo,
            'rest_diff': 0, 'rest_days_home': 2, 'rest_days_away': 2, # Defaults if schedule not linked
            'pace_mismatch': pace_m, 'eff_mismatch': eff_m, '3p_volatility': vol_3p,
            'h_eFG': h_u['eFG'], 'a_eFG': a_u['eFG'], 'h_tov': h_u['tov_pct'], 'a_tov': a_u['tov_pct']
        }])
        
        # Spread
        pred_spread = model.predict(row[features])[0]
        final_spread = pred_spread + net_impact
        
        diff_spread = final_spread - (spread * -1)
        edge_spread = abs(diff_spread)
        
        if diff_spread > 0: pick_spread = f"{home} {spread}"
        else: pick_spread = f"{away} +{abs(spread)}" if spread < 0 else f"{away} {spread}"
        
        t_line = f"{home} {final_spread*-1:.1f}" if final_spread > 0 else f"{away} {final_spread:.1f}"
        
        # Total
        pred_tot = model_tot.predict(row[features])[0]
        # Adjust total down if stars are out (less offense)
        final_tot = pred_tot - (h_loss + a_loss) 
        
        diff_tot = final_tot - vegas_total
        edge_tot = abs(diff_tot)
        
        pick_tot = f"Ov {vegas_total}" if diff_tot > 0 else f"Un {vegas_total}"
        
        # Traffic Lights
        light_s = "🟢" if edge_spread > 4.0 else "🟡" if edge_spread > 2.0 else "🔴"
        light_t = "🟢" if edge_tot > 5.0 else "🟡" if edge_tot > 2.5 else "🔴"
        
        analysis = explain_pick(row, home, away, net_impact, final_tot, vegas_total)
        
        print(f"{away} @ {home:<5} | {t_line:<10} | {spread:<6} | {pick_spread:<10} | {edge_spread:.1f} {light_s} | {final_tot:<8.1f} | {vegas_total:<6} | {pick_tot:<10} | {edge_tot:.1f} {light_t} | {analysis}")

if __name__ == "__main__":
    run_predictions()
