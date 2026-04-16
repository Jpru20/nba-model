import requests
from curl_cffi import requests as curl_requests

# 1. THE MASK: Monkey-patch standard requests to impersonate Google Chrome
class ChromeSession(curl_requests.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, impersonate="chrome")

requests.Session = ChromeSession

# 2. NOW import nba_api (it will unknowingly use the Chrome impersonator)
from nba_api.stats.endpoints import leaguegamelog

custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
}

# Your active proxy (removed the 30m timeout so it doesn't expire on us)
proxy_url = "http://jpru2188:223Bvevm62_country-us_state-newjersey@geo.iproyal.com:12321"

print("Pinging NBA.com with Proxy + Chrome TLS Spoofing...")

try:
    log = leaguegamelog.LeagueGameLog(
        season='2023-24', 
        player_or_team_abbreviation='T', 
        headers=custom_headers, 
        timeout=30,
        proxy=proxy_url
    )
    df = log.get_data_frames()[0]
    print(f"✅ SUCCESS! We bypassed Akamai's TLS Fingerprinting. Pulled {len(df)} games.")
except Exception as e:
    print(f"❌ FAILED: {e}")
