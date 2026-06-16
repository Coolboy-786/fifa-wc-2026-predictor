# ⚽ FIFA World Cup 2026 Match Predictor

A machine-learning app that predicts international football results and simulates the 2026 World Cup. It pairs an XGBoost win/draw/loss model with an Elo + recent-form feature pipeline, then runs a Monte Carlo simulation of the tournament to estimate each nation's title odds. Served as a FastAPI backend and a Streamlit web app.

> **Live demo:** _add your `https://<your-app>.streamlit.app` URL here once deployed_

---

## Features

- **Single-match predictor** — pick any two teams and get calibrated win / draw / loss probabilities.
- **Championship odds** — Monte Carlo simulation of the full 48-team tournament produces each team's probability of lifting the trophy.
- **Interactive UI** — Streamlit front end with team selectors and Altair charts.
- **REST API** — optional FastAPI service exposing the same predictions as JSON.

---

## How it works

**Data.** Historical international results from the community-maintained [`martj42/international_results`](https://github.com/martj42/international_results) dataset.

**Feature engineering.** Each match is described by leak-free, pre-match features computed only from games that happened *before* it:

| Feature | Description |
|---|---|
| `elo_diff` | Home Elo − Away Elo (with optional home-advantage term) |
| `home_elo`, `away_elo` | Current Elo ratings of each side |
| `neutral` | Whether the match is on neutral ground |
| `home_form`, `away_form` | Rolling recent-form scores |
| `form_diff` | Home form − Away form |

**Model.** An XGBoost classifier predicts the three-way outcome (home win / draw / away win), trained with a **time-based split** so the model is always validated on matches that occur after its training window — no temporal leakage. Approximate holdout performance: **~60% accuracy**, **~0.87 log loss**.

**Tournament simulation.** Groups are reconstructed via union-find, the top two of each group plus the eight best third-placed teams advance, and an Elo-seeded knockout bracket is resolved by the model over thousands of simulated tournaments to estimate title probabilities.

---

## Tech stack

Python · pandas · NumPy · XGBoost · scikit-learn · FastAPI · Uvicorn · Streamlit · Altair · joblib

---

## Project structure

```
.
├── app.py                     # Streamlit web app (imports features.py directly)
├── api.py                     # Optional FastAPI REST service
├── features.py                # Shared inference: loads artifacts, predict_match, odds
├── wc2026_artifacts.joblib    # Saved model + Elo ratings + form state + metadata
├── Fifa_wc_predictor.ipynb    # Notebook: data prep, training, simulation
├── requirements.txt
└── README.md
```

---

## Run locally

```bash
git clone https://github.com/Coolboy-786/fifa-wc-2026-predictor.git
cd fifa-wc-2026-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. It loads the saved model directly, so no backend is required.

### Optional: run the REST API

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/teams` | List of available teams |
| `GET` | `/predict?home=Argentina&away=Brazil` | Win/draw/loss probabilities |
| `GET` | `/odds?top=16` | Top-N championship odds |

---

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io), connected to this GitHub repo — pushing to `main` automatically rebuilds the live app. The Streamlit app imports `features.py` in-process, so only the single Streamlit service runs in the cloud (the FastAPI app is for local/standalone REST use).

---

## Notes & limitations

- Probabilities reflect historical patterns in international results; football is high-variance and upsets are common by design.
- Elo and form state are snapshotted at training time and bundled into the artifact; retrain the notebook to refresh them with newer results.
- Knockout ties are resolved probabilistically (draws redistributed to a decisive result), approximating extra time / penalties.
- Not affiliated with FIFA. For learning and entertainment, not betting.

---

## Acknowledgements

- Match data: [martj42/international_results](https://github.com/martj42/international_results)
- Built with XGBoost, FastAPI, and Streamlit.
