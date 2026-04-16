import requests
import psycopg2
from datetime import datetime
import pytz

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
DB_URL = "postgresql://neondb_owner:npg_fx3jXEOrYd4a@ep-jolly-sunset-a4ktuss0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"

CURRENT_SEASON = '2025-26'
SPORT_KEY = "basketball_nba"

# TEAM MAPPING (Must match Predictor exactly)
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

def get_scores():
    # daysFrom=2 gets games from yesterday and today
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores/?daysFrom=2&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return []

def update_database(games):
    if not games:
        print("No games found in API response.")
        return

    print(f"Connecting to DB to process {len(games)} games...")
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        updates_count = 0
        utc = pytz.utc
        eastern = pytz.timezone('US/Eastern')
        
        for game in games:
            # 1. Skip if no score data
            scores = game.get('scores')
            if not scores: 
                continue 

            home_team = game['home_team']
            away_team = game['away_team']

            # 2. Extract Scores
            h_score_val = next((s['score'] for s in scores if s['name'] == home_team), None)
            a_score_val = next((s['score'] for s in scores if s['name'] == away_team), None)

            if h_score_val is None or a_score_val is None:
                continue

            try:
                h_int = int(h_score_val)
                a_int = int(a_score_val)
            except ValueError:
                continue

            # 3. Reconstruct ID (Must match Predictor Logic: Season_Date_Away@Home)
            h_abbr = TEAM_MAP.get(home_team)
            a_abbr = TEAM_MAP.get(away_team)
            
            if not h_abbr or not a_abbr:
                continue
            
            # Parse Date from API to match ID format (YYYY-MM-DD)
            commence_str = game['commence_time']
            game_dt = datetime.strptime(commence_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc).astimezone(eastern)
            date_str = game_dt.strftime("%Y-%m-%d")

            game_id = f"{CURRENT_SEASON}_{date_str}_{a_abbr}@{h_abbr}"

            # 4. Update Database
            sql = """
                UPDATE "predictions_2.0"
                SET home_score = %s, 
                    away_score = %s, 
                    updated_at = NOW()
                WHERE game_id = %s
                AND (home_score IS DISTINCT FROM %s OR away_score IS DISTINCT FROM %s);
            """
            cur.execute(sql, (h_int, a_int, game_id, h_int, a_int))
            
            if cur.rowcount > 0:
                print(f"✅ UPDATED: {away_team} ({a_int}) @ {home_team} ({h_int})")
                updates_count += 1

        conn.commit()
        if updates_count > 0:
            print(f"--- SUCCESS: Updated {updates_count} NBA games. ---")
        else:
            print("--- INFO: No new score changes. ---")
            
        cur.close()

    except Exception as e:
        print(f"❌ DB Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    print(f"\n[{datetime.now()}] Starting NBA Score Update...")
    data = get_scores()
    update_database(data)
