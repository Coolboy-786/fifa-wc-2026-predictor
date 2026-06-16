from fastapi import FastAPI, HTTPException
import features as F

app = FastAPI(title="WC 2026 Predictor")

@app.get("/teams")
def teams():
    return {"teams": F.TEAMS}

@app.get("/predict")
def predict(home: str, away: str):
    if home not in F.ELOS or away not in F.ELOS:
        raise HTTPException(404, "unknown team")
    if home == away:
        raise HTTPException(400, "pick two different teams")
    return {"home": home, "away": away, "probs": F.predict_match(home, away)}

@app.get("/odds")
def odds(top: int = 16):
    return {"odds": [{"team": t, "prob": p} for t, p in F.championship_odds(top)]}
