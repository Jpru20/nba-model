import time
from datetime import datetime
import os
import psycopg2
from dotenv import load_dotenv

# --- 1. THE MASK: Monkey-patch standard requests to impersonate Google Chrome ---
import requests
from curl_cffi import requests as curl_requests

class ChromeSession(curl_requests.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, impersonate="chrome")

requests.Session = ChromeSession

# --- 2. APIs & Config ---
from nba_api.live.nba.endpoints import scoreboard, boxscore
from nba_api.stats.endpoints import leaguedashplayerstats

ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
SPORT_KEY = "basketball_nba"
DB_URL = "postgresql://neondb_owner:npg_fx3jXEOrYd4a@ep-jolly-sunset-a4ktuss0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"

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

# --- 3. DYNAMIC PIE DATABASE ---
def fetch_all_player_pies():
    """Fetches up-to-date PIE for EVERY active NBA player directly from the NBA API."""
    print("Downloading live PIE ratings for all NBA players...")
    pie_dict = {}
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense='Advanced'
        )
        data = stats.get_normalized_dict().get('LeagueDashPlayerStats', [])
        
        for player in data:
            name = player.get('PLAYER_NAME', 'Unknown')
            pie = player.get('PIE')
            if pie is not None:
                pie_dict[name] = float(pie)
                
        return pie_dict
    except Exception as e:
        print(f"  [!] API error fetching player PIEs: {e}")
        return {}

PLAYER_PIE_DB = fetch_all_player_pies()
LEAGUE_AVG_PIE = 0.100 

# --- 4. CORE FUNCTIONS ---
def fetch_todays_priors():
    """Fetches the pre-game projections and game_id from the Neon database."""
    priors = {}
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT home_team, away_team, predicted_total, consensus_spread, game_id 
            FROM "predictions_2.0" 
            WHERE league = 'NBA' 
            AND DATE(date) >= CURRENT_DATE - INTERVAL '2 days'
        """)
        rows = cur.fetchall()

        for home_full, away_full, p_tot_str, c_spr_str, game_id in rows:
            h_abbr = TEAM_MAP.get(home_full)
            a_abbr = TEAM_MAP.get(away_full)
            if not h_abbr or not a_abbr: continue

            try:
                tot = float(p_tot_str)
                parts = c_spr_str.split()
                favored_team = parts[0]
                spread_val = abs(float(parts[1])) 
                
                fav_score = (tot + spread_val) / 2
                dog_score = (tot - spread_val) / 2
                
                if favored_team == h_abbr:
                    h_score, a_score = fav_score, dog_score
                else:
                    h_score, a_score = dog_score, fav_score
                    
                priors[f"{a_abbr}@{h_abbr}"] = {
                    'pre_h_score': h_score,
                    'pre_a_score': a_score,
                    'game_id': game_id
                }
            except Exception as e:
                continue
                
        cur.close()
        conn.close()
        return priors
    except Exception as e:
        print(f"Database error fetching priors: {e}")
        return {}

def fetch_live_scores():
    """Fetches second-by-second live scores and clock data using bulletproof chaining."""
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        
        live_data = {}
        for game in games:
            status = game.get('gameStatus', 1)
            
            home_team = game.get('homeTeam', {}).get('teamTricode', 'UNK')
            away_team = game.get('awayTeam', {}).get('teamTricode', 'UNK')
            h_score = game.get('homeTeam', {}).get('score', 0)
            a_score = game.get('awayTeam', {}).get('score', 0)
            period = game.get('period', 1)
            
            clock = game.get('gameClock', '') 
            if clock.startswith('PT'):
                clock = clock.replace('PT', '').replace('M', ':').split('.')[0]
                if len(clock) == 4: clock = '0' + clock 
            
            live_data[f"{away_team}@{home_team}"] = {
                'nba_game_id': game.get('gameId'),
                'status': status,
                'period': period,
                'clock': clock if clock else "00:00",
                'home_score': h_score,
                'away_score': a_score
            }
        return live_data
    except Exception as e:
        print(f"Error fetching live scores: {e}")
        return {}

def fetch_live_odds():
    """Fetches in-game Vegas moneylines, spreads, and totals."""
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds'
    params = {
        'api_key': ODDS_API_KEY, 
        'regions': 'us', 
        'markets': 'h2h,spreads,totals', 
        'oddsFormat': 'american', 
        'bookmakers': 'betmgm,draftkings'
    }
    try:
        res = requests.get(url, params=params).json()
        odds_data = {}
        
        for game in res:
            h = TEAM_MAP.get(game.get('home_team'))
            a = TEAM_MAP.get(game.get('away_team'))
            if not h or not a: continue
            
            spread, total = "N/A", "N/A"
            h_ml, a_ml = "N/A", "N/A" 
            
            books = sorted(game.get('bookmakers', []), key=lambda x: 0 if x['key'] == 'betmgm' else 1)
            for bk in books:
                s_found, t_found, ml_found = None, None, False
                for m in bk.get('markets', []):
                    if m['key'] == 'spreads':
                        for o in m.get('outcomes', []):
                            if o['name'] == game.get('home_team'): s_found = o.get('point')
                    elif m['key'] == 'totals': 
                        if m.get('outcomes'): t_found = m['outcomes'][0].get('point')
                    elif m['key'] == 'h2h': 
                        ml_found = True
                        for o in m.get('outcomes', []):
                            if o['name'] == game.get('home_team'): h_ml = o.get('price')
                            if o['name'] == game.get('away_team'): a_ml = o.get('price')
                            
                if s_found is not None or ml_found:
                    spread = s_found if s_found is not None else "N/A"
                    total = t_found if t_found is not None else "N/A"
                    break
                    
            odds_data[f"{a}@{h}"] = {
                'live_spread': spread,
                'live_total': total,
                'home_ml': h_ml,
                'away_ml': a_ml
            }
        return odds_data
    except Exception as e:
        print(f"Error fetching live odds: {e}")
        return {}

def analyze_live_context(nba_game_id, period):
    """Scans the live box score for foul trouble and in-game injuries."""
    h_modifier = 0.0
    a_modifier = 0.0
    
    try:
        box = boxscore.BoxScore(nba_game_id)
        game_data = box.game.get_dict()
        
        home_players = game_data.get('homeTeam', {}).get('players', [])
        away_players = game_data.get('awayTeam', {}).get('players', [])
        
        def evaluate_team_context(players, current_period):
            team_penalty = 0.0
            for p in players:
                first_name = p.get('firstName', '')
                last_name = p.get('familyName', '')
                full_name = f"{first_name} {last_name}".strip()
                
                player_pie = PLAYER_PIE_DB.get(full_name, LEAGUE_AVG_PIE)
                pie_multiplier = player_pie / LEAGUE_AVG_PIE
                
                stats = p.get('statistics', {})
                fouls = stats.get('foulsPersonal', 0)
                mins_played = stats.get('minutesCalculated', '')
                
                if current_period == 1 and fouls >= 2:
                    team_penalty -= (1.5 * pie_multiplier)
                elif current_period == 2 and fouls >= 3:
                    team_penalty -= (1.5 * pie_multiplier)
                elif current_period == 3 and fouls >= 4:
                    team_penalty -= (1.0 * pie_multiplier)
                    
                try:
                    if mins_played:
                        m_int = int(mins_played.split('M')[0].replace('PT', ''))
                        
                        # THE FIX: Ensure m_int > 0 to ignore players inactive from the start!
                        # We also expanded the window to < 12 to catch stars who played a few minutes 
                        # in the 1st half but didn't return for the 2nd.
                        if current_period >= 3 and 0 < m_int < 12 and player_pie > 0.120:
                            team_penalty -= (3.0 * pie_multiplier) 
                except ValueError:
                    pass
                        
            return team_penalty

        h_modifier = evaluate_team_context(home_players, period)
        a_modifier = evaluate_team_context(away_players, period)
        
        return h_modifier, a_modifier
    except Exception as e:
        print(f"  [!] Boxscore PIE parsing error: {e}")
        return 0.0, 0.0

def calculate_live_projection(h_live_score, a_live_score, period, clock_str, pre_h_score, pre_a_score, h_modifier=0.0, a_modifier=0.0):
    """Calculates the live projected final score."""
    try:
        mins, secs = map(int, clock_str.split(':'))
    except ValueError:
        mins, secs = 0, 0
        
    if period <= 4:
        periods_left = 4 - period
        seconds_remaining = (periods_left * 12 * 60) + (mins * 60) + secs
        total_seconds = 48 * 60
    else:
        seconds_remaining = (mins * 60) + secs
        total_seconds = 5 * 60 
        
    seconds_elapsed = total_seconds - seconds_remaining
    if seconds_elapsed <= 0: seconds_elapsed = 1 

    h_pps_prior = pre_h_score / total_seconds
    a_pps_prior = pre_a_score / total_seconds
    
    h_pps_live = h_live_score / seconds_elapsed
    a_pps_live = a_live_score / seconds_elapsed
    
    live_weight = seconds_elapsed / total_seconds
    prior_weight = 1.0 - live_weight
    
    h_expected_remaining = seconds_remaining * ((h_pps_live * live_weight) + (h_pps_prior * prior_weight))
    a_expected_remaining = seconds_remaining * ((a_pps_live * live_weight) + (a_pps_prior * prior_weight))
    
    h_expected_remaining += h_modifier
    a_expected_remaining += a_modifier

    h_proj_final = h_live_score + h_expected_remaining
    a_proj_final = a_live_score + a_expected_remaining
    
    return float(round(h_proj_final, 1)), float(round(a_proj_final, 1))

def update_live_data_in_db(live_updates_payload):
    """Pushes the real-time live odds and dynamic status to the Neon database."""
    if not live_updates_payload: 
        return
        
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        update_query = """
            UPDATE "predictions_2.0"
            SET status = %s,
                live_clock = %s,
                live_home_score = %s,
                live_away_score = %s,
                live_home_ml = %s,
                live_away_ml = %s,
                live_vegas_spread = %s,
                live_vegas_total = %s,
                live_proj_margin = %s,
                live_proj_total = %s,
                updated_at = NOW()
            WHERE game_id = %s;
        """
        
        cur.executemany(update_query, live_updates_payload)
        conn.commit()
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  [DB] ❌ Database error updating live data: {e}")

def run_live_loop():
    print("Starting Live NBA Tracker with Quota Protection...")
    
    # --- THROTTLE CONFIG ---
    last_odds_fetch = 0
    cached_odds = {}
    ODDS_COOLDOWN_SECONDS = 180  

    # --- NEW: FINAL GAME TRACKER ---
    synced_finals = set()

    print("=" * 105)
    
    while True:
        try:
            db_priors = fetch_todays_priors()
            live_games = fetch_live_scores()
            
            # --- NEW: HIBERNATION SCHEDULER ---
            scheduled = [m for m, d in live_games.items() if d['status'] == 1]
            
            active_matchups = [m for m, d in live_games.items() if d['status'] in [2, 3] and m not in synced_finals]
            
            if len(active_matchups) > 0:
                current_time = datetime.now().strftime('%I:%M:%S %p')
                print(f"\n--- Update: {current_time} ---")
                print(f"{'MATCHUP':<10} | {'CLOCK':<12} | {'SCORE':<10} | {'LIVE ML':<11} | {'VEGAS S/T':<14} | {'OUR PROJ'}")
                print("-" * 105)
                
                current_time_sec = time.time()
                if (current_time_sec - last_odds_fetch) > ODDS_COOLDOWN_SECONDS:
                    cached_odds = fetch_live_odds()
                    last_odds_fetch = current_time_sec
                    print("  [API] Fetched fresh Vegas odds. Consumed 1 credit.")
                
                live_odds = cached_odds
                
                db_payload = []
                
                for matchup in active_matchups:
                    data = live_games[matchup]
                    odds = live_odds.get(matchup, {'live_spread': 'N/A', 'live_total': 'N/A', 'home_ml': 'N/A', 'away_ml': 'N/A'})
                    
                    prior = db_priors.get(matchup)
                    if prior:
                        pre_home_proj = prior['pre_h_score']
                        pre_away_proj = prior['pre_a_score']
                        game_id = prior['game_id']
                    else:
                        pre_home_proj, pre_away_proj, game_id = 110.0, 110.0, None

                    h_mod, a_mod = analyze_live_context(data['nba_game_id'], data['period'])

                    live_h_proj, live_a_proj = calculate_live_projection(
                        data['home_score'], data['away_score'], 
                        data['period'], data['clock'], 
                        pre_home_proj, pre_away_proj,
                        h_modifier=h_mod, a_modifier=a_mod
                    )
                    
                    live_proj_margin = float(live_a_proj - live_h_proj) 
                    live_proj_total = float(live_h_proj + live_a_proj)
                    
                    db_status = 'Final' if data['status'] == 3 else 'In Progress'
                    
                    if data['status'] == 3:
                        clock_display = "Final"
                        proj_display = "FINAL"
                        synced_finals.add(matchup) 
                    else:
                        q_str = f"Q{data['period']}" if data['period'] <= 4 else f"OT{data['period']-4}"
                        clock_display = f"{q_str} - {data['clock']}"
                        proj_display = f"H {live_proj_margin:+.1f} (Tot {live_proj_total:.1f})"
                        
                    score_display = f"{data['away_score']} - {data['home_score']}"
                    
                    # FORMAT FIX: Convert spread into a positive/negative string for the DB
                    spr = odds['live_spread']
                    spr_str = f"H {spr:+.1f}" if isinstance(spr, (int, float)) else str(spr)
                    spr_db_str = f"{spr:+.1f}" if isinstance(spr, (int, float)) else str(spr)
                    
                    a_ml, h_ml = odds['away_ml'], odds['home_ml']
                    a_ml_str = f"+{a_ml}" if isinstance(a_ml, (int, float)) and a_ml > 0 else str(a_ml) if a_ml != 'N/A' else "N/A"
                    h_ml_str = f"+{h_ml}" if isinstance(h_ml, (int, float)) and h_ml > 0 else str(h_ml) if h_ml != 'N/A' else "N/A"
                    ml_display = f"{a_ml_str}/{h_ml_str}"
                    
                    # FORMAT FIX: Convert live margin into a positive/negative string for the DB
                    live_proj_margin_db_str = f"{live_proj_margin:+.1f}"
                    
                    print(f"{matchup:<10} | {clock_display:<12} | {score_display:<10} | {ml_display:<11} | {spr_str:<7} {odds['live_total']:<6} | {proj_display}")
                    
                    if game_id:
                        db_payload.append((
                            db_status,
                            clock_display,
                            data['home_score'],
                            data['away_score'],
                            h_ml_str,
                            a_ml_str,
                            spr_db_str,                # <-- Now pushes exactly +3.5 
                            str(odds['live_total']),
                            live_proj_margin_db_str,   # <-- Now pushes exactly +6.3
                            live_proj_total,
                            game_id
                        ))
                
                if db_payload:
                    update_live_data_in_db(db_payload)
                    print(f"  [DB] Synced {len(db_payload)} live game updates to Neon.")
                else:
                    print(f"  [DB] No matching game_ids found in Neon. Skipped update.")
            
                # Active games are playing, poll every 30 seconds
                time.sleep(30)
                
            # --- NEW: HIBERNATION STATES ---
            else:
                if len(scheduled) > 0:
                    current_time = datetime.now().strftime('%I:%M:%S %p')
                    print(f"[{current_time}] 💤 No active games. Waiting for {len(scheduled)} scheduled game(s) to tip off. Hibernating for 5 minutes...")
                    time.sleep(300)
                else:
                    current_time = datetime.now().strftime('%I:%M:%S %p')
                    print(f"[{current_time}] 🌙 All games finished! Hibernating for 1 hour...")
                    synced_finals.clear() # Wipe memory clean for tomorrow
                    time.sleep(3600)
            
        except Exception as e:
            print(f"⚠️ Master Live Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_live_loop()
