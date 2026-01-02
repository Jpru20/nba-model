import joblib
import pandas as pd
import numpy as np
import requests
import scrape_nba_injuries 
from datetime import datetime, timedelta
import pytz
from nba_api.stats.endpoints import leaguegamelog

ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
CURRENT_SEASON = '2025-26' 

def get_live_odds():
    print("   [1/5] Fetching Live NBA Odds...")
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {'api_key': ODDS_API_KEY, 'regions': 'us', 'markets': 'spreads,totals', 'oddsFormat': 'american', 'bookmakers': 'betmgm,draftkings'}
    try:
        res = requests.get(url, params=params).json()
        matchups = []
        utc = pytz.utc
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        start_of_day = now_et.replace(hour=5, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(hours=24)

        team_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }

        for game in res:
            game_time = datetime.strptime(game['commence_time'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc).astimezone(eastern)
            if not (start_of_day <= game_time < end_of_day): continue 

            h, a = team_map.get(game['home_team']), team_map.get(game['away_team'])
            if not h or not a: continue
            
            spread, total = 0.0, 225.0
            books = sorted(game.get('bookmakers', []), key=lambda x: 0 if x['key'] == 'betmgm' else 1)
            for bk in books:
                s_found, t_found = None, None
                for m in bk['markets']:
                    if m['key'] == 'spreads':
                        for o in m['outcomes']:
                            if o['name'] == game['home_team']: s_found = o['point']
                    if m['key'] == 'totals': t_found = m['outcomes'][0]['point']
                if s_found is not None:
                    spread, total = s_found, t_found
                    break
            matchups.append({'home': h, 'away': a, 'spread': float(spread), 'total': float(total)})
        return matchups
    except: return []

def get_formatted_injuries():
    print("   [2/5] Reading Injury Data...")
    try:
        raw = scrape_nba_injuries.fetch_latest_injury_report_team_enriched()
        data = {}
        abbr_map = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA", "Washington Wizards": "WAS"
        }

        for team, players in raw.items():
            abbr = abbr_map.get(team, team[:3].upper())
            if abbr not in data: data[abbr] = {}
            for p in players:
                data[abbr][p['Player']] = {'Status': p['Status'].upper(), 'PIE': p.get('PIE', 0.08)}
        return data
    except Exception as e:
        print(f"Error reading injuries: {e}")
        return {}

def load_brains():
    print("   [3/5] Loading Models...")
    try:
        ms = joblib.load('nba_model_spread.pkl')
        mt = joblib.load('nba_model_total.pkl')
        elo = joblib.load('nba_elo_state.pkl')
        unit = pd.read_csv('nba_unit_stats.csv')
        vals = pd.read_csv('nba_player_values.csv')
        feats = joblib.load('nba_features.pkl')
        log = leaguegamelog.LeagueGameLog(season=CURRENT_SEASON, player_or_team_abbreviation='T').get_data_frames()[0]
        log['POSS'] = log['FGA'] + 0.44 * log['FTA'] + log['TOV'] - log['OREB']
        log['ORTG'] = (log['PTS'] / log['POSS']) * 100
        mc = log.groupby('TEAM_ABBREVIATION').agg(avg_pace=('POSS', 'mean'), avg_ortg=('ORTG', 'mean'))
        return ms, mt, elo, unit, vals, feats, mc, log
    except: return None

def adjust_total_for_modern_era(predicted_total, home, away, mc_stats, vol_3p):
    """
    Adjusts the raw model total to account for 2025 Pace and 3P Volume.
    """
    if home not in mc_stats.index or away not in mc_stats.index: return predicted_total
    
    h_pace = mc_stats.loc[home]['avg_pace']
    a_pace = mc_stats.loc[away]['avg_pace']
    avg_pace = (h_pace + a_pace) / 2
    
    # MODERN ERA FIX:
    # 1. Base Efficiency: Using 1.16 (approx 116.0 ORTG) instead of 1.12
    pace_derived_total = avg_pace * 1.16 * 2
    
    # 2. 3-Point Bonus: If 3P Volatility (volume) is high, add points
    # vol_3p is usually around 0.35-0.45 (3P Rate). 
    # If combined rate > 0.8 (super high), add ~4 points.
    three_pt_bonus = 0
    if vol_3p > 0.40: three_pt_bonus = 2.0
    if vol_3p > 0.45: three_pt_bonus = 4.0
    
    # 3. Weighted Blend: Increased Pace influence to 50% (was 30%)
    final_total = (predicted_total * 0.50) + (pace_derived_total * 0.50) + three_pt_bonus
    
    return final_total

def explain_pick(row, home, away, net_impact, pred_total, vegas_total, spread_edge):
    reasons = []
    
    # Elo
    elo_diff = row['elo_diff'].values[0]
    if elo_diff > 50: reasons.append(f"{home} Elo Adv")
    elif elo_diff < -50: reasons.append(f"{away} Elo Adv")
    
    # Rest
    rh = row['rest_days_home'].values[0]
    ra = row['rest_days_away'].values[0]
    if rh == 0 and ra > 0: reasons.append(f"⚠️ {home} B2B")
    if ra == 0 and rh > 0: reasons.append(f"⚠️ {away} B2B")
    
    # Stats
    efg_d = row['efg_mismatch'].values[0]
    if abs(efg_d) > 0.03:
        team = home if efg_d > 0 else away
        reasons.append(f"{team} Eff Edge")
        
    # Total
    total_diff = pred_total - vegas_total
    if total_diff > 5: reasons.append("High Pace/Shootout")
    elif total_diff < -5: reasons.append("Defensive Grind")
    
    # Injury
    if abs(net_impact) > 1.5:
        reasons.append(f"Inj Impact")

    return "; ".join(reasons) if reasons else "Stats Edge"

def run_consensus():
    print("\nInitializing NBA Consensus Model (Modern Scoring Update)...")
    brains = load_brains()
    if not brains: return
    model_s, model_t, elo, units, vals, feats, mc_stats, gamelog = brains
    
    matchups = get_live_odds()
    injuries = get_formatted_injuries()
    
    def get_impact(team):
        if team not in injuries: return 0.0, []
        loss, dets = 0.0, []
        weights = {'OUT': 1.0, 'DOUBTFUL': 0.75, 'QUESTIONABLE': 0.5}
        for p, d in injuries[team].items():
            if d['Status'] in weights:
                imp = min(d['PIE'] * 25 * weights[d['Status']], 7.0)
                if d['PIE'] < 0.07: imp *= 0.2
                loss += imp
                dets.append(f"{p} ({d['Status']})")
        return min(loss, 16.0), dets

    print("\n" + "="*165)
    print(f"{'MATCHUP':<12} | {'SCORE':<8} | {'XGB':<5} | {'SIM':<5} | {'CON':<5} | {'VEGAS':<6} | {'PICK':<12} | {'EDGE':<5} | {'CONF':<5} | {'TOT':<10} | {'ANALYSIS'}")
    print("="*165)
    
    teams_played = set()
    for m in matchups:
        home, away, spread, total = m['home'], m['away'], m['spread'], m['total']
        teams_played.update([home, away])
        
        h_u = units[units['TEAM_ABBREVIATION']==home].iloc[0] if home in units.values else None
        a_u = units[units['TEAM_ABBREVIATION']==away].iloc[0] if away in units.values else None
        
        vol_3p = ((h_u['fg3a_rate'] + a_u['fg3a_rate']) / 2) if h_u is not None else 0
        
        row = pd.DataFrame([{
            'elo_diff': elo.get(home, 1500) - elo.get(away, 1500),
            'home_elo': elo.get(home, 1500), 'away_elo': elo.get(away, 1500),
            'rest_diff': 0, 'rest_days_home': 2, 'rest_days_away': 2,
            'efg_mismatch': (h_u['eFG'] - a_u['eFG']) if h_u is not None else 0,
            'tov_mismatch': (a_u['TOV_pct'] - h_u['TOV_pct']) if h_u is not None else 0,
            'pace_mismatch': 0, 'combined_pace': 200, 'combined_efficiency': 220, '3p_volatility': vol_3p
        }])

        h_loss, h_d = get_impact(home)
        a_loss, a_d = get_impact(away)
        net_impact = a_loss - h_loss

        xgb = model_s.predict(row[feats])[0] + net_impact
        xgb_tot = model_t.predict(row[feats])[0] - ((h_loss+a_loss)*0.65)
        
        # Calculate Pace/Sim Margin
        sim_margin = 0
        if home in mc_stats.index and away in mc_stats.index:
            pace = (mc_stats.loc[home]['avg_pace'] + mc_stats.loc[away]['avg_pace']) / 2
            sim_margin = ((mc_stats.loc[home]['avg_ortg'] + 3.5) - mc_stats.loc[away]['avg_ortg']) / 100 * pace
            sim_margin += net_impact

        con_spr = (xgb + sim_margin) / 2
        
        # --- NEW TOTAL CALCULATION ---
        con_tot_raw = (xgb_tot + xgb_tot)/2 # Placeholder base
        con_tot = adjust_total_for_modern_era(con_tot_raw, home, away, mc_stats, vol_3p)
        
        h_score = int((con_tot + con_spr)/2)
        a_score = int((con_tot - con_spr)/2)
        
        vegas_margin = spread * -1 
        edge = abs(con_spr - vegas_margin)
        
        if con_spr > vegas_margin:
            p_side = f"{home} {spread}" if spread < 0 else f"{home} +{spread}"
        else:
            p_side = f"{away} +{abs(spread)}" if spread < 0 else f"{away} -{spread}"

        # TOT PICK LOGIC
        if con_tot > total:
            pick_t = f"Ov {total}"
        else:
            pick_t = f"Un {total}"

        conf = "HIGH" if edge > 5.5 else "MED" if edge > 2.0 else "LOW"
        
        base_analysis = explain_pick(row, home, away, net_impact, con_tot, total, edge)
        inj_str = f" (Inj: {home}-{h_loss:.1f}, {away}-{a_loss:.1f})"
        full_analysis = base_analysis + inj_str

        def s(val): return f"{val:.1f}"
        print(f"{away} @ {home:<5} | {a_score}-{h_score:<5} | {s(xgb):<5} | {s(sim_margin):<5} | {s(con_spr):<5} | {spread:<6} | {p_side:<12} | {edge:.1f}   | {conf:<5} | {pick_t:<10} | {full_analysis}")

    print("\nFULL INJURY REPORT (Playing Today)")
    print("-" * 40)
    for t in sorted(teams_played):
        if t in injuries and injuries[t]:
            loss = sum(d['PIE'] for d in injuries[t].values() if d['Status'] in ['OUT','DOUBTFUL','QUESTIONABLE'])
            print(f"{t} (Loss: {loss:.3f})")
            for p, d in injuries[t].items():
                print(f"  - {p:<20} {d['Status']:<12} {d['PIE']:.3f}")
            print("")
        else:
            print(f"{t} (No Injuries Reported)")
            print("")

if __name__ == "__main__":
    run_consensus()
