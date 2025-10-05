from fastapi import FastAPI
import pandas as pd
import requests
import joblib

app = FastAPI(title="F1 Singapore GP 2025 ML Predictor")

model = joblib.load("f1_model.pkl")

def get_current_driver_standings():
    url = "https://ergast.com/api/f1/current/driverStandings.json"
    data = requests.get(url).json()
    rows = []
    for d in data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]:
        name = f"{d['Driver']['givenName']} {d['Driver']['familyName']}"
        team = d["Constructors"][0]["name"]
        pts = float(d["points"])
        rows.append({"driver": name, "constructor": team, "season_points": pts})
    return pd.DataFrame(rows)

def get_singapore_history():
    url = "https://ergast.com/api/f1/circuits/marina_bay/results.json?limit=1000"
    data = requests.get(url).json()
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        for result in race["Results"]:
            driver = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
            constructor = result["Constructor"]["name"]
            points = int(result["points"])
            rows.append({"driver": driver, "constructor": constructor, "sgp_points": points})
    return pd.DataFrame(rows)

@app.get("/predict")
def predict():
    standings = get_current_driver_standings()
    history = get_singapore_history()

    avg_sgp = history.groupby("driver")["sgp_points"].mean().reset_index()
    df = standings.merge(avg_sgp, on="driver", how="left").fillna(0)

    team_avg = df.groupby("constructor")["season_points"].mean().to_dict()
    df["team_strength"] = df["constructor"].map(team_avg)

    X = df[["season_points", "sgp_points", "team_strength"]]
    df["predicted_finish"] = model.predict(X)

    df["probability"] = (1 / df["predicted_finish"])
    df["probability"] = df["probability"] / df["probability"].sum() * 100
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)

    podium = df.head(3)
    dark_horse = df.iloc[3]
    dnf = df.sample(1).iloc[0]

    caption = (
        "🏁 Singapore GP 2025 Predictions 🌴✨\n\n"
        f"🥇 {podium.iloc[0]['driver']} ({podium.iloc[0]['constructor']}) — {podium.iloc[0]['probability']:.1f}%\n"
        f"🥈 {podium.iloc[1]['driver']} ({podium.iloc[1]['constructor']}) — {podium.iloc[1]['probability']:.1f}%\n"
        f"🥉 {podium.iloc[2]['driver']} ({podium.iloc[2]['constructor']}) — {podium.iloc[2]['probability']:.1f}%\n\n"
        f"Dark Horse 🦅: {dark_horse['driver']} ({dark_horse['constructor']}) — {dark_horse['probability']:.1f}%\n"
        f"Possible DNF 😬: {dnf['driver']} ({dnf['constructor']}) — {dnf['probability']:.1f}%\n\n"
        "#SingaporeGP #F1 #Predictions #MarinaBay #Formula1 #AI #DataDriven"
    )

    return {"predictions": df.head(10).to_dict(orient="records"), "caption": caption}
