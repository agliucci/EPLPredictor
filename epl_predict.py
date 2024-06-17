import pandas as pd
import joblib

df = pd.read_csv("epl_results.csv") 

find_averages = df.drop(columns=["Date", "Time", "Referee", "HTR"])

print(find_averages["FTHG"].mean())


model = joblib.load('epl_decision_tree_model.pkl')
label_encoder_home = joblib.load('label_encoder_home.pkl')
label_encoder_away = joblib.load('label_encoder_away.pkl')
label_encoder_ftr = joblib.load('label_encoder_ftr.pkl')
label_encoder_htr = joblib.load('label_encoder_htr.pkl')

data_to_predict = {
    'HomeTeam': ['Man United'],
    'AwayTeam': ['Man City'],
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






new_data = pd.DataFrame(data_to_predict)


X_new = new_data[['HomeTeam_encoded', 'AwayTeam_encoded', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]


predictions = model.predict(X_new)


predicted_outcome = label_encoder_ftr.inverse_transform(predictions)[0]
print(f"Predicted Outcome: {predicted_outcome}")


def average(home_team, away_team):
    home = find_averages[(find_averages["HomeTeam"] == away_team) & (find_averages["AwayTeam"] == home_team)]
    away = find_averages[(find_averages["HomeTeam"] == home_team) & (find_averages["AwayTeam"] == away_team)]
    cols = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS","HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
    home_avg = home[cols].mean().to_dict()
    away_avg = away[cols].mean().to_dict()


    data_to_predict = {
        'HomeTeam': [home_team],
        'AwayTeam': [away_team],
        'FTHG': [home_avg['FTHG']],
        'FTAG': [away_avg['FTAG']],
        'HTHG': [home_avg['HTHG']],
        'HTAG': [away_avg['HTAG']],
        'HST': [home_avg['HST']],
        'AST': [away_avg['AST']],
        'HF': [home_avg['HF']],
        'AF': [away_avg['AF']],
        'HC': [home_avg['HC']],
        'AC': [away_avg['AC']],
        'HY': [home_avg['HY']],
        'AY': [away_avg['AY']],
        'HR': [home_avg['HR']],
        'AR': [away_avg['AR']],
    }
    return data_to_predict

def encode(home_team, away_team):
    data_to_predict['HomeTeam_encoded'] = label_encoder_home.transform([home_team])
    data_to_predict['AwayTeam_encoded'] = label_encoder_away.transform([away_team])
    return data_to_predict['HomeTeam_encoded'], data_to_predict['AwayTeam_encoded']

print(average("Man United", "Man City"))
print(encode("Man United", "Man City"))