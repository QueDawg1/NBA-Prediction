# Betting Line Converter
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from nba_api.stats.endpoints import leaguegamefinder

def advanced_nba_pipeline(target_season='2024-25'):
    print(f"Connecting to NBA.com to fetch {target_season} regular season stats...")
    
    # 1. Fetch raw logs using leaguegamefinder
    game_finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=target_season,
        season_type_nullable='Regular Season'
    )
    df = game_finder.get_data_frames()[0].copy()
    
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    
    # 2. ADVANCED TRACKING: Calculate Rest Days and Basic Game Pace
    # Estimate game possessions using the basic NBA.com framework: 
    # Possessions = FGA + 0.44 * FTA - OREB + TOV
    df['POSSESSIONS'] = df['FGA'] + (0.44 * df['FTA']) - df['OREB'] + df['TOV']
    
    print("Calculating team rest days and temporal metrics...")
    # Rest Days: Days between the current game and previous game date for that specific team
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days - 1
    # Cap rest days at 5 to prevent long off-season or All-Star break layout distortions
    df['REST_DAYS'] = df['REST_DAYS'].fillna(3).clip(upper=5) 
    
    # 3. Prevent Data Leakage: Calculate 5-Game Lagging Rolling Averages
    # All rolling features must represent status PRIOR to tip-off (shift by 1)
    df['ROLL_PTS'] = df.groupby('TEAM_ID')['PTS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['ROLL_PACE'] = df.groupby('TEAM_ID')['POSSESSIONS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # Extract opponent points to build defensive tracking parameters
    opp_df = df[['GAME_ID', 'TEAM_ID', 'PTS']].rename(columns={'TEAM_ID': 'OPP_ID', 'PTS': 'OPP_PTS'})
    
    # 4. Reconstruct Matchups (Join Home vs Away logs)
    home_df = df[df['MATCHUP'].str.contains('vs.')].copy()
    away_df = df[df['MATCHUP'].str.contains('@')].copy()
    
    matchups = pd.merge(home_df, away_df, on='GAME_ID', suffixes=('_HOME', '_AWAY'))
    
    # Map defensive traits (Points allowed rolling tracker)
    # Get what the home team typically surrenders to opponents
    matchups = pd.merge(matchups, opp_df, left_on=['GAME_ID', 'TEAM_ID_AWAY'], right_on=['GAME_ID', 'OPP_ID']).rename(columns={'OPP_PTS': 'ACTUAL_PTS_ALLOWED_BY_AWAY'})
    matchups = pd.merge(matchups, opp_df, left_on=['GAME_ID', 'TEAM_ID_HOME'], right_on=['GAME_ID', 'OPP_ID']).rename(columns={'OPP_PTS': 'ACTUAL_PTS_ALLOWED_BY_HOME'})
    
    matchups['HOME_ROLL_DEF'] = matchups.groupby('TEAM_ID_HOME')['ACTUAL_PTS_ALLOWED_BY_HOME'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    matchups['AWAY_ROLL_DEF'] = matchups.groupby('TEAM_ID_AWAY')['ACTUAL_PTS_ALLOWED_BY_AWAY'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    matchups = matchups.dropna(subset=['HOME_ROLL_DEF', 'AWAY_ROLL_DEF', 'ROLL_PTS_HOME', 'ROLL_PTS_AWAY'])
    
    # 5. Model Architecture & Splitting
    feature_cols = [
        'ROLL_PTS_HOME', 'HOME_ROLL_DEF', 'REST_DAYS_HOME', 'ROLL_PACE_HOME',
        'ROLL_PTS_AWAY', 'AWAY_ROLL_DEF', 'REST_DAYS_AWAY', 'ROLL_PACE_AWAY'
    ]
    
    X = matchups[feature_cols]
    y_home = matchups['PTS_HOME']
    y_away = matchups['PTS_AWAY']
    
    split = int(len(matchups) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_home, y_test_home = y_home.iloc[:split], y_home.iloc[split:]
    y_train_away, y_test_away = y_away.iloc[:split], y_away.iloc[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_home = LinearRegression().fit(X_train_scaled, y_train_home)
    model_away = LinearRegression().fit(X_train_scaled, y_train_away)
    
    return model_home, model_away, scaler, feature_cols

# 6. VEGAS OVER/UNDER AND SPREAD TRANSLATION ENGINE
def evaluate_vegas_value(home_pred, away_pred, vegas_total, vegas_spread):
    """
    Translates individual projected points to compare directly with sports book betting lines.
    
    vegas_spread: Note as negative for Home favorite (e.g. -4.5 means Home is favored by 4.5)
    """
    model_total = home_pred + away_pred
    model_spread = away_pred - home_pred  # Negative means Home team wins out
    
    print("\n" + "="*50)
    print("      NBA GAME SCORING PROJECTION & LINE ANALYSIS      ")
    print("="*50)
    print(f"Projected Final Score: Home {home_pred:.1f} | Away {away_pred:.1f}")
    print(f"Model Predicted Game Total: {model_total:.1f} points")
    print(f"Model Predicted Spread:     Home {'+' if model_spread > 0 else ''}{model_spread:.1f}")
    print("-"*50)
    print(f"Vegas Book Lines:           Total O/U: {vegas_total} | Spread: {vegas_spread}")
    print("-"*50)
    
    # Over/Under Value Analysis
    total_delta = model_total - vegas_total
    if abs(total_delta) >= 3.5:
        ou_rec = f"🔥 BET {'OVER' if total_delta > 0 else 'UNDER'} (Edge: {abs(total_delta):.1f} pts)"
    else:
        ou_rec = "NO VALUE - Line is efficiently placed."
        
    # Spread Value Analysis (Vegas vs Model discrepancy)
    spread_delta = vegas_spread - model_spread
    if abs(spread_delta) >= 2.5:
        spread_rec = f"🔥 BET {'HOME' if spread_delta > 0 else 'AWAY'} SPREAD (Edge: {abs(spread_delta):.1f} pts)"
    else:
        spread_rec = "NO VALUE - Line matches structural indicators."
        
    print(f"Over/Under Recommendation:  {ou_rec}")
    print(f"Point Spread Recommendation: {spread_rec}")
    print("="*50 + "\n")

# Run Pipeline Training
m_home, m_away, pipeline_scaler, features = advanced_nba_pipeline('2023-24')

# Inference Run Example: 
# Input: [HOME_OFF, HOME_DEF, HOME_REST, HOME_PACE, AWAY_OFF, AWAY_DEF, AWAY_REST, AWAY_PACE]
mock_matchup = np.array([[116.2, 114.5, 0, 99.1, 110.4, 118.1, 2, 102.4]]) # Home on B2B, fast Away team
mock_scaled = pipeline_scaler.transform(mock_matchup)

pred_pts_home = m_home.predict(mock_scaled)[0]
pred_pts_away = m_away.predict(mock_scaled)[0]

# Simulate a real upcoming Vegas board item: Over/Under 222.5, Home favored by 5.5
evaluate_vegas_value(
    home_pred=pred_pts_home, 
    away_pred=pred_pts_away, 
    vegas_total=222.5, 
    vegas_spread=-5.5
)
