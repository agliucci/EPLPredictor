import joblib


model = joblib.load('epl_decision_tree_model.pkl')
label_encoder_home = joblib.load('label_encoder_home.pkl')
label_encoder_away = joblib.load('label_encoder_away.pkl')
label_encoder_ftr = joblib.load('label_encoder_ftr.pkl')
label_encoder_htr = joblib.load('label_encoder_htr.pkl')

data_to_predict = {
    'HomeTeam': ['Man United'],
    'AwayTeam': ['Chelsea'],
    'FTHG': [2],
    'FTAG': [1],
    'HTHG': [1],
    'HTAG': [0],
    'HST': [6],
    'AST': [3],
    'HF': [10],
    'AF': [8],
    'HC': [5],
    'AC': [2],
    'HY': [1],
    'AY': [2],
    'HR': [0],
    'AR': [0]
}

data_to_predict['HomeTeam_encoded'] = label_encoder_home.transform(data_to_predict['HomeTeam'])
data_to_predict['AwayTeam_encoded'] = label_encoder_away.transform(data_to_predict['AwayTeam'])

import pandas as pd

# Create DataFrame from dictionary
new_data = pd.DataFrame(data_to_predict)

# Select relevant features
X_new = new_data[['HomeTeam_encoded', 'AwayTeam_encoded', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]

# Make prediction
predictions = model.predict(X_new)

# Inverse transform predictions to get original labels
predicted_outcome = label_encoder_ftr.inverse_transform(predictions)[0]
print(f"Predicted Outcome: {predicted_outcome}")

