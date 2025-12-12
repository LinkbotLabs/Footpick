import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime

st.title("⚽ Premier League Match Predictor App")
st.markdown("Built with free data from [football-data.org](https://www.football-data.org). Predicts match outcomes using machine learning!")

# API Base
BASE_URL = "https://api.football-data.org/v4"

# Cache data fetching
@st.cache_data(ttl=600)  # Refresh every 10 minutes
def fetch_matches(season_year):
    url = f"{BASE_URL}/competitions/PL/matches?season={season_year}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json()['matches'])
    else:
        st.error("Error fetching data. Try again later.")
        return pd.DataFrame()

# Fetch historical data (last few seasons for training)
st.sidebar.header("Model Training")
seasons = [2020, 2021, 2022, 2023, 2024]  # Add more if needed
historical_data = pd.DataFrame()

for year in seasons:
    with st.spinner(f"Fetching season {year}/{year+1}..."):
        df = fetch_matches(year)
        if not df.empty:
            historical_data = pd.concat([historical_data, df], ignore_index=True)

if historical_data.empty:
    st.stop()

# Preprocess historical data
historical_data = historical_data[historical_data['status'] == 'FINISHED'].copy()
historical_data['homeGoals'] = historical_data['score'].apply(lambda x: x['fullTime']['home'] if x['fullTime']['home'] is not None else 0)
historical_data['awayGoals'] = historical_data['score'].apply(lambda x: x['fullTime']['away'] if x['fullTime']['away'] is not None else 0)

# Create target: 0 = Home Win, 1 = Draw, 2 = Away Win
historical_data['result'] = np.where(historical_data['homeGoals'] > historical_data['awayGoals'], 0,
                           np.where(historical_data['homeGoals'] == historical_data['awayGoals'], 1, 2))

# Simple features: goal difference, form not included yet for simplicity (can expand)
# Encode teams
le_home = LabelEncoder()
le_away = LabelEncoder()
historical_data['homeTeamId'] = le_home.fit_transform(historical_data['homeTeam'].apply(lambda x: x['name']))
historical_data['awayTeamId'] = le_away.fit_transform(historical_data['awayTeam'].apply(lambda x: x['name']))

features = ['homeTeamId', 'awayTeamId']
X = historical_data[features]
y = historical_data['result']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(multi_class='multinomial', max_iter=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.sidebar.success(f"Model trained on {len(historical_data)} matches | Test Accuracy: {accuracy:.2%}")

# Current season fixtures
current_year = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
current_fixtures = fetch_matches(current_year)

if current_fixtures.empty:
    st.stop()

# Show upcoming / live matches
st.header(f"Current Season {current_year}/{current_year+1} Fixtures")
live_or_upcoming = current_fixtures[current_fixtures['status'].isin(['SCHEDULED', 'LIVE', 'IN_PLAY', 'PAUSED'])]
if not live_or_upcoming.empty:
    display_df = live_or_upcoming[['utcDate', 'homeTeam', 'awayTeam', 'status', 'score']].copy()
    display_df['Date'] = pd.to_datetime(display_df['utcDate']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['Home'] = display_df['homeTeam'].apply(lambda x: x['name'])
    display_df['Away'] = display_df['awayTeam'].apply(lambda x: x['name'])
    display_df['Score'] = display_df['score'].apply(lambda x: f"{x['fullTime']['home'] or 0}-{x['fullTime']['away'] or 0}" if x['fullTime']['home'] is not None else "-")
    st.dataframe(display_df[['Date', 'Home', 'Away', 'Score', 'status']].rename(columns={'status': 'Status'}))

# Prediction section
st.header("Predict a Match")
home_team = st.selectbox("Select Home Team", sorted(le_home.classes_))
away_team = st.selectbox("Select Away Team", sorted(le_away.classes_))

if st.button("Predict Outcome"):
    if home_team == away_team:
        st.warning("Teams can't be the same!")
    else:
        home_id = le_home.transform([home_team])[0]
        away_id = le_away.transform([away_team])[0]
        pred_prob = model.predict_proba([[home_id, away_id]])[0]
        pred_class = np.argmax(pred_prob)
        
        outcomes = ["Home Win", "Draw", "Away Win"]
        st.success(f"**Prediction: {outcomes[pred_class]}**")
        st.progress(max(pred_prob))
        st.write(f"{home_team} Win: {pred_prob[0]:.1%}")
        st.write(f"Draw: {pred_prob[1]:.1%}")
        st.write(f"{away_team} Win: {pred_prob[2]:.1%}")
        
        # Simple detailed info (expandable)
        with st.expander("More Details (Form & Head-to-Head)"):
            st.info("Advanced form/head-to-head coming soon! Model based purely on team strength for now.")

st.markdown("---")
st.caption("Note: This is a basic ML model (~55-60% accuracy typical for football). Live scores update on refresh. For production, add more features like form streaks, xG, etc.")
