from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from tqdm import tqdm

final_log = pd.read_csv('final_log.csv')
final_log['GAME_DATE'] = pd.to_datetime(final_log['GAME_DATE'])

best_lgbm = joblib.load('best_lgbm.joblib')
lstm_model = load_model('lstm_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

def get_sorted_players(team_abbr, final_log):
    team_players = final_log[
        (final_log['TEAM_ABBR'] == team_abbr) & (~final_log['PLAYER_NAME'].isnull())
    ][['Player_ID', 'PLAYER_NAME', 'PTS_career']].drop_duplicates()
    team_players = team_players.sort_values(by='PTS_career', ascending=False)
    return team_players['PLAYER_NAME'].unique().tolist()

def prepare_selected_game_features(team1_name, team2_name, team1_players, team2_players, final_log, scaler, train_window=3):
    team1_features = aggregate_team_features(team1_name, team1_players, final_log, train_window)
    team2_features = aggregate_team_features(team2_name, team2_players, final_log, train_window)

    if not team1_features or not team2_features:
        raise ValueError("Insufficient data for one or both teams.")

    interaction_features = {}
    for feature in team1_features.keys():
        if feature not in ['Offense_Defense_Interaction']:
            interaction_features[f"{feature}_diff"] = team1_features[feature] - team2_features[feature]

    features = {f"team1_{k}": v for k, v in team1_features.items()}
    features.update({f"team2_{k}": v for k, v in team2_features.items()})
    features.update(interaction_features)

    features_df = pd.DataFrame([features])

    if hasattr(scaler, 'feature_names_in_'):
        for col in scaler.feature_names_in_:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[scaler.feature_names_in_]

    print("Aligned features for prediction:", list(features_df.columns))

    return features_df


def aggregate_team_features(team_name, player_names, final_log, train_window=3):
    player_ids = final_log[
        (final_log['TEAM_ABBR'] == team_name) & (final_log['PLAYER_NAME'].isin(player_names))
    ]['Player_ID'].unique()

    recent_games = final_log[
        (final_log['TEAM_ABBR'] == team_name) & (final_log['Player_ID'].isin(player_ids))
    ].sort_values(by='GAME_DATE', ascending=False).groupby('Player_ID').head(train_window)

    if recent_games.empty:
        return None

    team_features = {
        **{f"{col}_avg": recent_games[col].mean(skipna=True) for col in recent_games.columns if col.endswith("_game")},
        **{f"{col}_sum": recent_games[col].sum(skipna=True) for col in recent_games.columns if col.endswith("_game")},
        **{f"{col}_avg": recent_games[col].mean(skipna=True) for col in recent_games.columns if col.endswith("_career")},
        **{f"{col}_sum": recent_games[col].sum(skipna=True) for col in recent_games.columns if col.endswith("_career")},
        "Offense_Defense_Interaction": (recent_games['PTS_game'].sum(skipna=True) + recent_games['AST_game'].sum(skipna=True)) *
                                       (recent_games['REB_game'].sum(skipna=True) + recent_games['STL_game'].sum(skipna=True))
    }
    return team_features

def predict_outcome(team1_name, team2_name, team1_players, team2_players, final_log, lgbm_model, lstm_model, scaler):
    features_df = prepare_selected_game_features(team1_name, team2_name, team1_players, team2_players, final_log, scaler)

    if features_df is None or features_df.empty:
        return "Insufficient data to make a prediction."

    features_scaled = scaler.transform(features_df)

    lgbm_probs = lgbm_model.predict_proba(features_scaled)[:, 1]

    lstm_input = features_scaled.reshape(-1, features_scaled.shape[1], 1)
    lstm_probs = lstm_model.predict(lstm_input).flatten()

    ensemble_probs = (lgbm_probs + lstm_probs) / 2
    return "Team 1 Will Win" if ensemble_probs > 0.5 else "Team 2 Will Win"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_players', methods=['POST'])
def get_players():
    data = request.json
    team_abbr = data.get('team_abbr', '').strip().upper()
    players = get_sorted_players(team_abbr, final_log)
    return jsonify(players)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    team1_name = data.get('team1_name', '').strip().upper()
    team2_name = data.get('team2_name', '').strip().upper()
    team1_players = data.get('team1_players', [])
    team2_players = data.get('team2_players', [])

    if team1_name == team2_name:
        return jsonify({'error': "Teams must be different!"}), 400

    try:
        prediction = predict_outcome(
            team1_name, team2_name, team1_players, team2_players, final_log, best_lgbm, lstm_model, scaler
        )
        return jsonify({'prediction': prediction})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
