from datetime import datetime
import json
import os
from pathlib import Path
import time
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import feedparser
from google import genai
from nba_api.stats.endpoints import leaguedashteamstats, scoreboardv2
from nba_api.stats.static import teams

load_dotenv()

NBA_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/131.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
}

NBA_TIMEOUT = 60
MAX_RETRIES = 3


def _get_gemini_api_key() -> str | None:
    if os.environ.get('GEMINI_API_KEY'):
        return os.environ['GEMINI_API_KEY']
    try:
        import streamlit as st
        return st.secrets.get('GEMINI_API_KEY')
    except Exception:
        return None


api_key = _get_gemini_api_key()
if not api_key:
    raise RuntimeError(
        'Missing GEMINI_API_KEY. Set it in your environment/.env or Streamlit Secrets.'
    )

client = genai.Client(api_key=api_key)
MODEL_NAME = 'gemini-3.1-flash-lite'


def _call_nba_endpoint(factory_fn, label: str, retries: int = MAX_RETRIES):
    """Calls an nba_api endpoint factory with retries and exponential backoff."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return factory_fn()
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                sleep_seconds = 2**attempt
                print(
                    f'[Warning] {label} attempt {attempt}/{retries} failed ({exc}). '
                    f'Retrying in {sleep_seconds}s...'
                )
                time.sleep(sleep_seconds)
    raise last_err


def get_tonights_matchups(game_date_et: str | None = None):
    """Gets NBA matchups for the given date (defaults to current ET date)."""
    if game_date_et is None:
        game_date_et = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

    all_teams = teams.get_teams()
    id_to_team = {t['id']: t for t in all_teams}

    try:
        board = _call_nba_endpoint(
            lambda: scoreboardv2.ScoreboardV2(
                game_date=game_date_et,
                headers=NBA_HEADERS,
                timeout=NBA_TIMEOUT,
            ),
            label=f'ScoreboardV2({game_date_et})',
        )
        games_df = board.game_header.get_data_frame()
    except Exception as exc:
        print(f'[Error] Could not fetch scoreboard from stats.nba.com: {exc}')
        return []

    matchups = []
    for _, row in games_df.iterrows():
        home = id_to_team.get(int(row['HOME_TEAM_ID']))
        away = id_to_team.get(int(row['VISITOR_TEAM_ID']))
        if home and away:
            matchups.append({
                'game_date': game_date_et,
                'home_abbr': home['abbreviation'],
                'home_name': home['full_name'],
                'away_abbr': away['abbreviation'],
                'away_name': away['full_name'],
                'status': row['GAME_STATUS_TEXT'],
            })

    return matchups


def fetch_news_for_playing_teams(playing_abbrs, articles_per_team=5):
    """Fetches recent news headlines for the playing teams."""
    abbr_to_team = {t['abbreviation']: t for t in teams.get_teams()}
    headlines_by_team = {}

    for abbr in sorted(playing_abbrs):
        team = abbr_to_team.get(abbr)
        if not team:
            continue

        query = quote_plus(f'"{team["full_name"]}" NBA')
        rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'

        feed = feedparser.parse(rss_url)
        entries = getattr(feed, 'entries', [])[:articles_per_team]
        headlines_by_team[abbr] = [e.get('title', '') for e in entries if e.get('title')]
        time.sleep(0.5)

    return headlines_by_team


def fetch_team_stats(fallback_json_path: str | Path = 'team_stats_fallback.json'):
    """Gets per-game team stats, falling back to a local JSON file if stats.nba.com times out."""
    id_to_abbr = {t['id']: t['abbreviation'] for t in teams.get_teams()}

    try:
        endpoint = _call_nba_endpoint(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                per_mode_detailed='PerGame',
                headers=NBA_HEADERS,
                timeout=NBA_TIMEOUT,
            ),
            label='LeagueDashTeamStats',
        )
        df = endpoint.get_data_frames()[0]

        stats = {}
        for _, row in df.iterrows():
            abbr = id_to_abbr.get(int(row['TEAM_ID']))
            if abbr:
                stats[abbr] = {
                    'record': f"{int(row['W'])}-{int(row['L'])}",
                    'win_pct': round(float(row['W_PCT']), 3),
                    'ppg': round(float(row['PTS']), 1),
                    'plus_minus': round(float(row['PLUS_MINUS']), 1),
                }

        # Cache latest successful stats locally as a fallback
        try:
            Path(fallback_json_path).write_text(json.dumps(stats, indent=2), encoding='utf-8')
        except OSError:
            pass

        return stats

    except Exception as exc:
        print(f'[Warning] LeagueDashTeamStats failed ({exc}). Checking local fallback...')
        fallback_file = Path(fallback_json_path)
        if fallback_file.exists():
            return json.loads(fallback_file.read_text(encoding='utf-8'))
        return {}


def predict_matchups(matchups, stats, headlines_by_team):
    """Generates predictions and sentiment scores using Gemini."""
    blocks = []
    for m in matchups:
        home = m['home_abbr']
        away = m['away_abbr']

        away_news = '\n'.join(f'  - {h}' for h in headlines_by_team.get(away, [])) or '  - No recent headlines'
        home_news = '\n'.join(f'  - {h}' for h in headlines_by_team.get(home, [])) or '  - No recent headlines'

        blocks.append(
            f"Matchup: {away} @ {home} ({m['status']})\n"
            f'  {away} (Away) stats: {stats.get(away, {})}\n'
            f'  {away} headlines:\n{away_news}\n'
            f'  {home} (Home) stats: {stats.get(home, {})}\n'
            f'  {home} headlines:\n{home_news}'
        )

    prompt = (
        'Look at the stats and recent news headlines for each NBA matchup below.\n'
        'For every team playing:\n'
        '1. Give a sentiment score from -1.0 (very negative, e.g. key injury or slump) '
        'to +1.0 (very positive, e.g. winning streak or healthy return).\n'
        '2. Predict who wins each game, give rough win probabilities, and explain why briefly.\n\n'
        + '\n\n---\n\n'.join(blocks)
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


if __name__ == '__main__':
    matchups = get_tonights_matchups()
    if not matchups:
        print('No games tonight or unable to load matchups.')
        raise SystemExit(0)

    playing_teams = {m['home_abbr'] for m in matchups} | {m['away_abbr'] for m in matchups}
    print(f"Date: {matchups[0]['game_date']} ET | Games: {len(matchups)}")
    print(f'Teams playing ({len(playing_teams)}): {sorted(playing_teams)}\n')

    headlines_by_team = fetch_news_for_playing_teams(playing_teams, articles_per_team=5)
    stats = fetch_team_stats()

    output = predict_matchups(matchups, stats, headlines_by_team)
    print(output)