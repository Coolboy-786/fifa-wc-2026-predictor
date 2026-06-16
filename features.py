import joblib, numpy as np, pandas as pd

_A = joblib.load("wc2026_artifacts.joblib")
MODEL, ELOS, FORM = _A["model"], _A["elos"], _A["form"]
FEATURES, HOME_ADV = _A["FEATURES"], _A["HOME_ADV"]
TEAMS, CHAMPIONS = _A["teams"], _A["champions"]

def predict_match(home, away, neutral=True):
    he = ELOS.get(home, 1500.0)
    ae = ELOS.get(away, 1500.0)
    hf = np.mean(FORM[home]) if FORM.get(home) else 0.0
    af = np.mean(FORM[away]) if FORM.get(away) else 0.0
    elo_diff = (he + (0 if neutral else HOME_ADV)) - ae
    x = pd.DataFrame([[elo_diff, he, ae, int(neutral), hf, af, hf - af]], columns=FEATURES)
    p = MODEL.predict_proba(x)[0]
    return {"home_win": round(float(p[0]), 3),
            "draw":     round(float(p[1]), 3),
            "away_win": round(float(p[2]), 3)}

def championship_odds(top=None):
    total = sum(CHAMPIONS.values()) or 1
    items = sorted(((t, c / total) for t, c in CHAMPIONS.items()), key=lambda x: -x[1])
    return items[:top] if top else items
