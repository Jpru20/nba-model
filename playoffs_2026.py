import joblib
import pandas as pd
import numpy as np
import scrape_nba_injuries 
from datetime import datetime, timedelta
import pytz
import os
import psycopg2

# --- 1. THE MASK: Monkey-patch standard requests to impersonate Google Chrome ---
import requests
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv

class ChromeSession(curl_requests.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, impersonate="chrome")

requests.Session = ChromeSession

# --- 2. NOW import nba_api (it will unknowingly use the Chrome impersonator) ---
from nba_api.stats.endpoints import leaguegamelog

# --- 3. Load Proxy Credentials ---
load_dotenv()
PROXY_URL = os.getenv("NBA_PROXY_URL")

custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
}

# ==========================================
# 1. CONFIGURATION
# ==========================================
ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
CURRENT_SEASON = '2025-26' 
DB_URL = "postgresql://neondb_owner:npg_fx3jXEOrYd4a@ep-jolly-sunset-a4ktuss0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
SPORT_KEY = "basketball_nba"  

# ==========================================
# 2. TEAM MAPPING (Global)
# ==========================================
TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}
ABBR_TO_FULL = {v: k for k, v in TEAM_MAP.items()}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_live_odds():
    print("    [1/5] Fetching Live NBA Odds...")
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

        for game in res:
            commence_str = game['commence_time']
            game_time = datetime.strptime(commence_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc).astimezone(eastern)
            
            if not (start_of_day <= game_time): continue 
            if game_time < now_et: continue 

            h, a = TEAM_MAP.get(game['home_team']), TEAM_MAP.get(game['away_team'])
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
            
            matchups.append({
                'home': h, 
                'away': a, 
                'spread': float(spread), 
                'total': float(total),
                'date_iso': game_time.isoformat() 
            })
        return matchups
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return []

def get_formatted_injuries():
    print("    [2/5] Reading Injury Data...")
    try:
        raw = scrape_nba_injuries.fetch_latest_injury_report_team_enriched()
        # --- NEW: Save raw injuries to Database ---
        save_injuries_to_db(raw)
        # ------------------------------------------
        data = {}
        for team, players in raw.items():
            abbr = TEAM_MAP.get(team, team[:3].upper())
            if abbr not in data: data[abbr] = {}
            for p in players:
                data[abbr][p['Player']] = {'Status': p['Status'].upper(), 'PIE': p.get('PIE', 0.08)}
        return data
    except Exception as e:
        print(f"Error reading injuries: {e}")
        return {}

def calculate_days_rest(team_abbr, current_game_date, gamelog):
    try:
        team_games = gamelog[gamelog['TEAM_ABBREVIATION'] == team_abbr].copy()
        team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
        current_date = pd.to_datetime(current_game_date).tz_localize(None) 
        past_games = team_games[team_games['GAME_DATE'] < current_date]
        
        if past_games.empty: return 3 
            
        last_game_date = past_games['GAME_DATE'].max()
        delta = (current_date - last_game_date).days
        return min(delta, 3)
    except Exception as e:
        return 2

def load_brains():
    print("    [3/5] Loading Models...")
    try:
        ms = joblib.load('nba_model_spread.pkl')
        mt = joblib.load('nba_model_total.pkl')
        elo = joblib.load('nba_elo_state.pkl')
        unit = pd.read_csv('nba_unit_stats.csv')
        vals = pd.read_csv('nba_player_values.csv')
        feats = joblib.load('nba_features.pkl')
        
        log = leaguegamelog.LeagueGameLog(
            season=CURRENT_SEASON, 
            player_or_team_abbreviation='T',
            headers=custom_headers,
            proxy=PROXY_URL,
            timeout=30
        ).get_data_frames()[0]
        
        log['POSS'] = log['FGA'] + 0.44 * log['FTA'] + log['TOV'] - log['OREB']
        log['ORTG'] = (log['PTS'] / log['POSS']) * 100
        mc = log.groupby('TEAM_ABBREVIATION').agg(avg_pace=('POSS', 'mean'), avg_ortg=('ORTG', 'mean'))
        return ms, mt, elo, unit, vals, feats, mc, log
    except Exception as e:
        print(f"Error loading brains: {e}")
        return None

def adjust_total_for_modern_era(predicted_total, home, away, mc_stats, vol_3p):
    if home not in mc_stats.index or away not in mc_stats.index: return predicted_total
    
    h_pace = mc_stats.loc[home]['avg_pace']
    a_pace = mc_stats.loc[away]['avg_pace']
    avg_pace = (h_pace + a_pace) / 2
    
    pace_derived_total = avg_pace * 1.14 * 2
    
    three_pt_bonus = 0
    if vol_3p > 0.40: three_pt_bonus = 2.0
    if vol_3p > 0.45: three_pt_bonus = 4.0
    
    final_total = (predicted_total * 0.50) + (pace_derived_total * 0.50) + three_pt_bonus
    
    return final_total

def explain_pick(row, home, away, net_impact, pred_total, vegas_total, spread_edge):
    reasons = []
    
    elo_diff = row['elo_diff'].values[0]
    if elo_diff > 50: reasons.append(f"{home} Elo Adv")
    elif elo_diff < -50: reasons.append(f"{away} Elo Adv")
    
    rh = row['rest_days_home'].values[0]
    ra = row['rest_days_away'].values[0]
    if rh == 0 and ra > 0: reasons.append(f"⚠️ {home} B2B")
    if ra == 0 and rh > 0: reasons.append(f"⚠️ {away} B2B")
    
    efg_d = row['efg_mismatch'].values[0]
    if abs(efg_d) > 0.03:
        team = home if efg_d > 0 else away
        reasons.append(f"{team} Eff Edge")
        
    total_diff = pred_total - vegas_total
    if total_diff > 5: reasons.append("High Pace/Shootout")
    elif total_diff < -5: reasons.append("Defensive Grind")
    
    if abs(net_impact) > 1.5:
        reasons.append(f"Inj Impact")

    return "; ".join(reasons) if reasons else "Stats Edge"

def save_injuries_to_db(raw_injuries):
    if not raw_injuries: return
    
    print("   [DB] Saving today's injury report to database...")
    eastern = pytz.timezone('US/Eastern')
    report_date = datetime.now(eastern).strftime('%Y-%m-%d')
    
    db_payload = []
    for team, players in raw_injuries.items():
        for p in players:
            db_payload.append((
                report_date,
                p.get('Team', team),
                p.get('Player'),
                p.get('Status'),
                p.get('Reason', ''),
                p.get('PIE', 0.08)
            ))
            
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO nba_injuries (report_date, team, player, status, reason, pie, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (report_date, player) 
        DO UPDATE SET 
            status = EXCLUDED.status,
            reason = EXCLUDED.reason,
            pie = EXCLUDED.pie,
            updated_at = NOW();
        """
        cur.executemany(insert_query, db_payload)
        conn.commit()
        print(f"   [DB] ✅ Successfully saved {len(db_payload)} injury records for {report_date}.")
        cur.close()
    except Exception as e:
        print(f"   [DB] ❌ Database Error saving injuries: {e}")
    finally:
        if conn is not None:
            conn.close()
# ==========================================
# 4. DB SAVING FUNCTION
# ==========================================
def save_predictions_to_db(predictions):
    if not predictions:
        print("\n    [DB] No predictions to save.")
        return

    print(f"\n    [DB] Connecting to Neon Database to save {len(predictions)} records...")
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        insert_query = """
        INSERT INTO "predictions_2.0" (
            league, home_team, away_team, date, 
            predicted_winner, predicted_spread, predicted_total, 
            consensus_spread, spread, total, confidence, 
            sport_id, game_id, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, 
            %s, %s, %s, 
            %s, %s, %s, %s, 
            %s, %s, NOW(), NOW()
        ) ON CONFLICT (game_id) DO UPDATE SET
            home_team = EXCLUDED.home_team,
            away_team = EXCLUDED.away_team,
            predicted_winner = EXCLUDED.predicted_winner,
            predicted_spread = EXCLUDED.predicted_spread,
            predicted_total = EXCLUDED.predicted_total,
            consensus_spread = EXCLUDED.consensus_spread,
            spread = EXCLUDED.spread,
            total = EXCLUDED.total,
            confidence = EXCLUDED.confidence,
            updated_at = NOW();
        """

        conf_map = {"LOW": "low", "MED": "medium", "HIGH": "high"}

        for p in predictions:
            date_str = p['date_iso'][:10] 
            game_id_str = f"{CURRENT_SEASON}_{date_str}_{p['away']}@{p['home']}"
            conf_val = conf_map.get(p['confidence'], "low")
            con_val = p.get('con_spread_val', "N/A")

            cur.execute(insert_query, (
                'NBA',              # league
                p['home_full'],     # home_team
                p['away_full'],     # away_team
                p['date_iso'],      # date
                p['pick_team'],     # predicted_winner
                p['pick_text'],     # predicted_spread
                str(p['con_total']),# predicted_total
                str(con_val),       # consensus_spread 
                str(p['vegas_spread']),   # spread
                str(p['vegas_total']),    # total
                conf_val,           # confidence
                3,                  # sport_id
                game_id_str         # game_id
            ))

        conn.commit()
        print("    [DB] ✅ Successfully saved/updated NBA predictions.")
        cur.close()
    except Exception as e:
        print(f"    [DB] ❌ Database Error: {e}")
    finally:
        if conn is not None:
            conn.close()

# --- HELPER: UPDATE FINAL SCORES ---
def update_final_scores():
    print("\n--- CHECKING FOR COMPLETED GAMES (LAST 3 DAYS) ---")
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores'
    params = {
        'api_key': ODDS_API_KEY,
        'daysFrom': 3,
        'dateFormat': 'iso'
    }

    try:
        res = requests.get(url, params=params).json()
        updates = []
        
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        for game in res:
            if not game.get('completed', False):
                continue
            
            scores = game.get('scores', [])
            if not scores: continue
            
            h_score, a_score = 0, 0
            for s in scores:
                if s['name'] == game['home_team']: h_score = int(s['score'])
                if s['name'] == game['away_team']: a_score = int(s['score'])

            date_str = game['commence_time'][:10]
            away_abbr = TEAM_MAP.get(game['away_team'])
            home_abbr = TEAM_MAP.get(game['home_team'])
            if not away_abbr or not home_abbr:
                continue

            game_id = f"{CURRENT_SEASON}_{date_str}_{away_abbr}@{home_abbr}"
            updates.append((h_score, a_score, game_id))

        if updates:
            print(f"    -> Found {len(updates)} finished games. Updating DB...")
            update_query = """
                UPDATE "predictions_2.0"
                SET home_score = %s, 
                    away_score = %s, 
                    status = 'FINAL',
                    updated_at = NOW()
                WHERE game_id = %s;
            """
            cur.executemany(update_query, updates)
            conn.commit()
            print("    -> Scores updated successfully.")
        else:
            print("    -> No new finished games found.")
            
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error updating scores: {e}")

# ==========================================
# 5. MAIN LOGIC
# ==========================================
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
        weights = {'OUT': 1.0, 'DOUBTFUL': 0.75, 'QUESTIONABLE': 0.25} 
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
    
    db_payload = [] 
    teams_played = set()

    for m in matchups:
        home, away, spread, total = m['home'], m['away'], m['spread'], m['total']
        teams_played.update([home, away])

        h_u = units[units['TEAM_ABBREVIATION']==home].iloc[0] if home in units.values else None
        a_u = units[units['TEAM_ABBREVIATION']==away].iloc[0] if away in units.values else None

        if h_u is None or a_u is None: continue

        vol_3p = ((h_u['fg3a_rate'] + a_u['fg3a_rate']) / 2)

        current_date = m['date_iso'] 
        # PLAYOFF TWEAK: Playoff scheduling is standardized. Neutralize rest advantage.
        rest_home = 2 
        rest_away = 2
        

        # --- DUAL-FEATURE ROW CONSTRUCTION ---
        row = pd.DataFrame([{
            'elo_diff': elo.get(home, 1500) - elo.get(away, 1500),
            'home_elo': elo.get(home, 1500), 'away_elo': elo.get(away, 1500),
            'rest_diff': rest_home - rest_away,
            'rest_days_home': rest_home, 
            'rest_days_away': rest_away,
            
            # Baseline Class (Season Averages)
            'efg_mismatch': h_u['eFG'] - a_u['eFG'],
            'tov_mismatch': a_u['TOV_pct'] - h_u['TOV_pct'],
            
            # Current Form (14-game EMA)
            'ema_efg_mismatch': h_u['ema_eFG'] - a_u['ema_eFG'],
            'ema_tov_mismatch': a_u['ema_TOV_pct'] - h_u['ema_TOV_pct'],
            
            'pace_mismatch': h_u['POSS'] - a_u['POSS'],
            'combined_pace': h_u['POSS'] + a_u['POSS'], 
            'combined_efficiency': h_u['off_rtg'] + a_u['off_rtg'], 
            '3p_volatility': vol_3p
        }])

        h_loss, h_d = get_impact(home)
        a_loss, a_d = get_impact(away)
        net_impact = a_loss - h_loss
# ---------------------------------------------------------
        # [NEW] THE TANK TAX: Manual Penalty for Overvalued Teams
        # ---------------------------------------------------------
        # This subtracts points from their projected score.
        # Format: 'TEAM': Points_To_Penalize
        # IF BKN is constantly favored but losing, give them a -3.0 or -4.0 tax.
        TANK_TAX = {
            'BKN': 4.5,  # The model loves them, so we hit them hard
            'UTA': 3.5,  # Consistent loser penalty
            'POR': 3.0,  # Tanking team
        }

        # Apply Penalty to Home Team (Lowers their predicted margin)
        if home in TANK_TAX:
            net_impact -= TANK_TAX[home]
            # Optional: Add to "Analysis" string later so you know why
            
        # Apply Penalty to Away Team (Increases Home margin / Lowers Away score)
        if away in TANK_TAX:
            net_impact += TANK_TAX[away]
        # ---------------------------------------------------------
        xgb = model_s.predict(row[feats])[0] + net_impact
        xgb_tot = model_t.predict(row[feats])[0] - ((h_loss+a_loss)*0.65)

        # ---------------------------------------------------------
        # [NEW] PLAYOFF CONTEXT MATRIX
        # ---------------------------------------------------------
        SERIES_STATE = {
            # Manually tag today's spots here before running the script
            # Example: 'LAL': 'HOME_DOWN_0_2'
        }

        PLAYOFF_MODIFIERS = {
            'HOME_DOWN_0_2': 3.0,  
            'HOME_DOWN_0_1': 2.9,  
            'HOME_DOWN_1_2': 2.5,  
        }
        # ---------------------------------------------------------

        sim_margin = 0
        if home in mc_stats.index and away in mc_stats.index:
            # PLAYOFF TWEAK: 0.965 pace reduction multiplier for half-court style
            pace = ((mc_stats.loc[home]['avg_pace'] + mc_stats.loc[away]['avg_pace']) / 2) * 0.965
            
            # PLAYOFF TWEAK: Amplified home court advantage from +2.3 to +3.5
            ortg_diff = (mc_stats.loc[home]['avg_ortg'] + 3.5) - mc_stats.loc[away]['avg_ortg']
            
            # PLAYOFF TWEAK: Shifted to 14-game EMA to capture in-series adjustments
            efg_impact = row['ema_efg_mismatch'].values[0] * 200 
            tov_impact = row['ema_tov_mismatch'].values[0] * 125
            elo_impact = row['elo_diff'].values[0] / 25
            
            raw_margin = (ortg_diff * 0.3) + (efg_impact * 0.3) + (tov_impact * 0.2) + (elo_impact * 0.2)
            sim_margin = (raw_margin / 100) * pace
            sim_margin += net_impact

            # PLAYOFF TWEAK: Apply the Context Matrix
            if home in SERIES_STATE:
                spot = SERIES_STATE[home]
                if spot in PLAYOFF_MODIFIERS:
                    sim_margin += PLAYOFF_MODIFIERS[spot]

        con_spr = (xgb * 0.60) + (sim_margin * 0.40)

        con_tot_raw = (xgb_tot + xgb_tot)/2 
        con_tot = adjust_total_for_modern_era(con_tot_raw, home, away, mc_stats, vol_3p)

        h_score = int((con_tot + con_spr)/2)
        a_score = int((con_tot - con_spr)/2)

        vegas_margin = spread * -1 
        edge = abs(con_spr - vegas_margin)

        home_full = ABBR_TO_FULL.get(home, home)
        away_full = ABBR_TO_FULL.get(away, away)

        # 1. NEW: Determine the Straight-Up Winner (For the DB)
        straight_up_winner_full = home_full if con_spr > 0 else away_full

        if con_spr > vegas_margin:
            con_val_str = f"{home} -{abs(con_spr):.1f}"
            pick_text_full = f"{home_full} {spread}" if spread < 0 else f"{home_full} +{spread}"
            p_side = f"{home} {spread}" if spread < 0 else f"{home} +{spread}"
        else:
            con_val_str = f"{away} -{abs(con_spr):.1f}"
            pick_text_full = f"{away_full} +{abs(spread)}" if spread < 0 else f"{away_full} -{spread}"
            p_side = f"{away} +{abs(spread)}" if spread < 0 else f"{away} -{spread}"

        if con_tot > total:
            pick_t = f"Ov {total}"
        else:
            pick_t = f"Un {total}"

        conf = "HIGH" if edge > 5.5 else "MED" if edge > 2.0 else "LOW"

        base_analysis = explain_pick(row, home, away, net_impact, con_tot, total, edge)
        inj_str = f" (Inj: {home}-{h_loss:.1f}, {away}-{a_loss:.1f})"
        full_analysis = base_analysis + inj_str

        db_payload.append({
            "home": home, "away": away,
            "home_full": home_full, "away_full": away_full,
            "date_iso": m.get('date_iso'),
            "pick_team": straight_up_winner_full, 
            "pick_text": pick_text_full,
            "con_total": (h_score + a_score), 
            "con_spread_val": con_val_str, 
            "vegas_spread": spread,
            "vegas_total": total,
            "confidence": conf
        })
        def s(val): return f"{val:.1f}"
        print(f"{away} @ {home:<5} | {a_score}-{h_score:<5} | {s(xgb):<5} | {s(sim_margin):<5} | {s(con_spr):<5} | {spread:<6} | {p_side:<12} | {edge:.1f}    | {conf:<5} | {pick_t:<10} | {full_analysis}")

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

    save_predictions_to_db(db_payload)

# ==========================================
# 6. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    update_final_scores()
    run_consensus()
