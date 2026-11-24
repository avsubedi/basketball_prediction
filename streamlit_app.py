from datetime import datetime
import os
import time
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import feedparser
from google import genai
from nba_api.stats.endpoints import leaguedashteamstats, scoreboardv2
from nba_api.stats.static import teams

load_dotenv()


# Load AI and decide version of AI
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
MODEL_NAME = 'gemini-2.5-flash'


def get_tonights_matchups(game_date_et=None):
    """
    gets nba matchup for the night given
    """
    
    if game_date_et is None:
        game_date_et = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

    all_teams = teams.get_teams()
    id_to_team = {t['id']: t for t in all_teams}

    board = scoreboardv2.ScoreboardV2(game_date=game_date_et)
    games_df = board.game_header.get_data_frame()

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
    """
    feteches news for the playing teams given
    """
    
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
        time.sleep(0.3)

    return headlines_by_team


def fetch_team_stats():
    """
    gets team stats for the playing teams
    """
    
    id_to_abbr = {t['id']: t['abbreviation'] for t in teams.get_teams()}
    df = leaguedashteamstats.LeagueDashTeamStats(per_mode_detailed='PerGame').get_data_frames()[0]

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

    return stats


def predict_matchups(matchups, stats, headlines_by_team):
    """
    this gets gemini to predict based on given information 
    """
    
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
        print('No games tonight.')
        raise SystemExit(0)

    playing_teams = {m['home_abbr'] for m in matchups} | {m['away_abbr'] for m in matchups}
    print(f"Date: {matchups[0]['game_date']} ET | Games: {len(matchups)}")
    print(f'Teams playing ({len(playing_teams)}): {sorted(playing_teams)}\n')

    headlines_by_team = fetch_news_for_playing_teams(playing_teams, articles_per_team=5)
    stats = fetch_team_stats()

    output = predict_matchups(matchups, stats, headlines_by_team)
    print(output)
