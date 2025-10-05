import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import joblib

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
            year = int(race["season"])
            position = int(result["position"])
            points = int(result["points"])
            rows.append({"year": year, "driver": driver, "constructor": constructor,
                         "position": position, "sgp_points": points})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Fetching data...")
    standings = get_current_driver_standings()
    history = get_singapore_history()

    avg_sgp = history.groupby("driver")["sgp_points"].mean().reset_index()
    df = standings.merge(avg_sgp, on="driver", how="left").fillna(0)

    team_avg = df.groupby("constructor")["season_points"].mean().to_dict()
    df["team_strength"] = df["constructor"].map(team_avg)

    df["expected_finish"] = 25 - (df["season_points"] * 0.3 + df["sgp_points"] * 0.7)

    X = df[["season_points", "sgp_points", "team_strength"]]
    y = df["expected_finish"]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "f1_model.pkl")

    print("✅ Model trained and saved as f1_model.pkl")
