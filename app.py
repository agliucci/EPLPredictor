from flask import Flask, render_template, request
import joblib
import pandas as pd
from waitress import serve

df = pd.read_csv("epl_results.csv") 

find_averages = df.drop(columns=["Date", "Time", "Referee", "HTR"])

app = Flask(__name__)

model = joblib.load('epl_decision_tree_model.pkl')
label_encoder_home = joblib.load('label_encoder_home.pkl')
label_encoder_away = joblib.load('label_encoder_away.pkl')
label_encoder_ftr = joblib.load('label_encoder_ftr.pkl')
label_encoder_htr = joblib.load('label_encoder_htr.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text="", prediction_made=False)
   

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        home_team = home_team.title()
        away_team = away_team.title()


        def home_average(home_team):
            home = find_averages[(find_averages["HomeTeam"] == home_team)]
            cols = ["FTHG", "FTAG", "HTHG", "HTAG", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
            home_avg = list(home[cols].mean())
            return home_avg

        def away_average(away_team):
            away = find_averages[(find_averages["AwayTeam"] == away_team)]
            cols = ["FTHG", "FTAG", "HTHG", "HTAG","HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
            away_avg = list(away[cols].mean())
            return away_avg

        def home_encode(home_team):
            home_encoded = label_encoder_home.fit_transform([home_team])
            return home_encoded

        def away_encode(away_team):
            away_encoded = label_encoder_away.fit_transform([away_team])
            return away_encoded
        

        data_to_predict = {
            'HomeTeam': [home_team],
            'AwayTeam': [away_team],
            'FTHG': [home_average(home_team)[0]],
            'FTAG': [away_average(away_team)[1]],
            'HTHG': [home_average(home_team)[2]],
            'HTAG': [away_average(away_team)[3]],
            'HST': [home_average(home_team)[4]],
            'AST': [away_average(away_team)[5]],
            'HF': [home_average(home_team)[6]],
            'AF': [away_average(away_team)[7]],
            'HC': [home_average(home_team)[8]],
            'AC': [away_average(away_team)[9]],
            'HY': [home_average(home_team)[10]],
            'AY': [away_average(away_team)[11]],
            'HR': [home_average(home_team)[12]],
            'AR': [away_average(away_team)[13]]
        }

        data_to_predict['HomeTeam_encoded'] = label_encoder_home.fit_transform(data_to_predict['HomeTeam'])
        data_to_predict["AwayTeam_encoded"] = label_encoder_away.fit_transform(data_to_predict['AwayTeam'])

        new_data = pd.DataFrame(data_to_predict)

        data_to_predict["HomeTeam"] = home_encode(home_team)
        data_to_predict["AwayTeam_encoded"] = away_encode(away_team)
        X_new = new_data[['HomeTeam_encoded', 'AwayTeam_encoded', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]

        predictions = model.predict(X_new)

        predicted_outcome = label_encoder_ftr.inverse_transform(predictions)[0]
        if predicted_outcome == "H":
            predicted_outcome = home_team + " Win"
        else:
            predicted_outcome = away_team + " Win"
        if home_team not in df["HomeTeam"].values or away_team not in df["AwayTeam"].values or home_team == away_team or len(home_team) == 0 or len(away_team) == 0:
            return render_template('index.html', prediction_text="Please enter valid teams", prediction_made=True)
        else:
            return render_template('index.html', prediction_text=f"Predicted Outcome: {predicted_outcome}", prediction_made=True)

    return render_template('index.html')

    

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
