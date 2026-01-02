import requests
import fitz  # PyMuPDF
import json
import os
import re
import time
import pandas as pd
import unicodedata
from typing import List, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

INJURY_REPORT_INDEX_URL = "https://official.nba.com/nba-injury-report-2025-26-season/"
FALLBACK_INDEX_URLS = [
    "https://official.nba.com/nba-injury-report-2024-25-season/",
    "https://official.nba.com/nba-injury-report-2023-24-season/",
]

DEFAULT_OUTPUT_FILE = "data/nba_injuries.json"
PLAYER_VALUES_CSV = "nba_player_values.csv"
DEFAULT_PIE_FALLBACK = 0.08

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

# MAP KEYWORDS TO FULL OFFICIAL NAMES
KEYWORD_TO_FULL_NAME = {
    "Hawks": "Atlanta Hawks", "Atlanta": "Atlanta Hawks",
    "Celtics": "Boston Celtics", "Boston": "Boston Celtics",
    "Nets": "Brooklyn Nets", "Brooklyn": "Brooklyn Nets",
    "Hornets": "Charlotte Hornets", "Charlotte": "Charlotte Hornets",
    "Bulls": "Chicago Bulls", "Chicago": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers", "Cleveland": "Cleveland Cavaliers",
    "Mavericks": "Dallas Mavericks", "Dallas": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets", "Denver": "Denver Nuggets",
    "Pistons": "Detroit Pistons", "Detroit": "Detroit Pistons",
    "Warriors": "Golden State Warriors", "Golden State": "Golden State Warriors",
    "Rockets": "Houston Rockets", "Houston": "Houston Rockets",
    "Pacers": "Indiana Pacers", "Indiana": "Indiana Pacers",
    "Clippers": "LA Clippers", "LA Clippers": "LA Clippers",
    "Lakers": "Los Angeles Lakers", "Los Angeles": "Los Angeles Lakers",
    "Grizzlies": "Memphis Grizzlies", "Memphis": "Memphis Grizzlies",
    "Heat": "Miami Heat", "Miami": "Miami Heat",
    "Bucks": "Milwaukee Bucks", "Milwaukee": "Milwaukee Bucks",
    "Timberwolves": "Minnesota Timberwolves", "Minnesota": "Minnesota Timberwolves",
    "Pelicans": "New Orleans Pelicans", "New Orleans": "New Orleans Pelicans",
    "Knicks": "New York Knicks", "New York": "New York Knicks",
    "Thunder": "Oklahoma City Thunder", "Oklahoma City": "Oklahoma City Thunder",
    "Magic": "Orlando Magic", "Orlando": "Orlando Magic",
    "76ers": "Philadelphia 76ers", "Philadelphia": "Philadelphia 76ers",
    "Suns": "Phoenix Suns", "Phoenix": "Phoenix Suns",
    "Trail Blazers": "Portland Trail Blazers", "Portland": "Portland Trail Blazers",
    "Kings": "Sacramento Kings", "Sacramento": "Sacramento Kings",
    "Spurs": "San Antonio Spurs", "San Antonio": "San Antonio Spurs",
    "Raptors": "Toronto Raptors", "Toronto": "Toronto Raptors",
    "Jazz": "Utah Jazz", "Utah": "Utah Jazz",
    "Wizards": "Washington Wizards", "Washington": "Washington Wizards"
}

STATUS_WORDS = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
MATCHUP_RE = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")

# UPDATED PLAYER REGEX: Does NOT require the name to be at the start of the line (^ removed)
PLAYER_RE = re.compile(r"([A-Za-z\.\-']+,\s[A-Za-z\.\-']+)")

class InjuryReportError(Exception):
    pass

# =========================
# NETWORK HELPERS
# =========================

def make_session(headers: Dict[str, str]) -> requests.Session:
    s = requests.Session()
    s.headers.update(headers)
    return s

def request_with_retries(session, method, url, **kwargs):
    for attempt in range(1, 4):
        try:
            resp = session.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception:
            time.sleep(1.5 ** attempt)
    raise Exception(f"Network failed: {url}")

def get_latest_injury_report_url():
    session = make_session(HEADERS)
    # Just grab the index page and find the first PDF link
    try:
        html = request_with_retries(session, "GET", INJURY_REPORT_INDEX_URL, timeout=30).text
        pdfs = re.findall(r"https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_[^\"'\s]+\.pdf", html, flags=re.IGNORECASE)
        if pdfs: return pdfs[-1]
    except: pass
    
    # Fallback URLs
    for url in FALLBACK_INDEX_URLS:
        try:
            html = request_with_retries(session, "GET", url, timeout=30).text
            pdfs = re.findall(r"https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_[^\"'\s]+\.pdf", html, flags=re.IGNORECASE)
            if pdfs: return pdfs[-1]
        except: continue
        
    raise InjuryReportError("Could not locate PDF link.")

def download_pdf_bytes(url):
    print(f"--- Downloading Injury Report from {url} ---")
    return request_with_retries(make_session(HEADERS), "GET", url, timeout=45).content

def parse_pdf_name(pdf_name):
    if "," in pdf_name:
        last, first = pdf_name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return pdf_name

def _clean_lines(text):
    out = []
    # We do NOT filter 'Game Date' lines because in this format, data exists on that same line!
    for raw in text.splitlines():
        s = raw.strip()
        if not s: continue
        if "Injury Report:" in s or "Page " in s: continue
        out.append(s)
    return out

def _identify_team_in_line(line):
    """
    Scans the line to see if a full team name or keyword is present.
    """
    for kw, full_name in KEYWORD_TO_FULL_NAME.items():
        # Check if keyword exists in line with word boundaries
        if re.search(r'\b' + re.escape(kw) + r'\b', line, re.IGNORECASE):
            return full_name
    return None

def _parse_injury_pdf(doc: fitz.Document) -> List[Dict[str, Optional[str]]]:
    records = []
    current_team = "Unknown Team"
    
    # We sort=True to handle the dense layout order
    for page in doc:
        text = page.get_text("text", sort=True)
        lines = _clean_lines(text)
        
        for line in lines:
            # 1. Update Current Team if found in line
            # (In condensed format, Team is on the same line as the first player)
            found_team = _identify_team_in_line(line)
            if found_team:
                current_team = found_team
            
            # 2. Find Player Name (Anywhere in line)
            player_match = PLAYER_RE.search(line)
            if player_match:
                raw_name = player_match.group(1)
                
                # Validation: Name shouldn't be too long or contain "Injury"
                if len(raw_name) < 40 and "Injury" not in raw_name and "Game" not in raw_name:
                    player_name = parse_pdf_name(raw_name)
                    
                    # 3. Find Status (Anywhere in line)
                    status_found = None
                    for sw in STATUS_WORDS:
                        # Ensure we match the word "Out" but not "Without"
                        if re.search(r'\b' + re.escape(sw) + r'\b', line, re.IGNORECASE):
                            status_found = sw
                            break
                    
                    if status_found:
                        records.append({
                            "Team": current_team,
                            "Player": player_name,
                            "Status": status_found,
                            "Reason": line # Store full line as reason context
                        })

    return records

def parse_injury_report_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try: return _parse_injury_pdf(doc)
    finally: doc.close()

# =========================
# LOCAL ENRICHMENT
# =========================

def _normalize_player_name(name: Optional[str]) -> str:
    if not name:
        return ""
    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().replace(".", "").strip()


def get_local_player_values():
    print(f"--- Loading Player Values from {PLAYER_VALUES_CSV} ---")
    if not os.path.exists(PLAYER_VALUES_CSV):
        print("⚠️ CSV not found.")
        return {}
    try:
        df = pd.read_csv(PLAYER_VALUES_CSV)
        val_col = 'PIE' if 'PIE' in df.columns else 'value_per_game'
        if val_col not in df.columns: return {}
        
        lookup = {}
        for _, row in df.iterrows():
            name = _normalize_player_name(row.get("PLAYER_NAME"))
            if not name: continue
            val = float(row.get(val_col, DEFAULT_PIE_FALLBACK))
            if val > 1.0: val /= 150.0
            if val > 0.35: val = 0.08 # Sanity Check
            lookup[name] = val
        print(f"✅ Loaded values for {len(lookup)} players.")
        return lookup
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return {}

def fetch_latest_injury_report_team_enriched(include_pie=True):
    url = get_latest_injury_report_url()
    pdf_bytes = download_pdf_bytes(url)
    records = parse_injury_report_pdf_bytes(pdf_bytes)
    
    val_lookup = get_local_player_values()
    grouped = defaultdict(list)
    
    for r in records:
        name_key = _normalize_player_name(r["Player"])
        r["PIE"] = val_lookup.get(name_key, DEFAULT_PIE_FALLBACK)
        grouped[r["Team"]].append(r)
        
    return dict(grouped)

def save_injuries_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    print("🧪 Starting Scraper (Dense Layout Fix)...")
    try:
        data = fetch_latest_injury_report_team_enriched()
        count = sum(len(v) for v in data.values())
        print(f"✅ Success. Fetched injuries for {len(data)} teams, {count} total players.")
        save_injuries_json(data, DEFAULT_OUTPUT_FILE)
        
        # Verify
        first = next(iter(data.keys()), None)
        if first:
            print(f"\nSample Data ({first}):")
            print(json.dumps(data[first][:1], indent=2))
            
    except Exception as e:
        print(f"❌ Error: {e}")
