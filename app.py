import requests, pandas as pd, streamlit as st, altair as alt

API = "http://localhost:8000"
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")
st.title("⚽ FIFA World Cup 2026 Predictor")

@st.cache_data(ttl=300)
def get(path, **params):
    return requests.get(f"{API}{path}", params=params, timeout=15).json()

teams = get("/teams")["teams"]
st.subheader("Match predictor")
c1, c2 = st.columns(2)
home = c1.selectbox("Home", teams, index=0)
away = c2.selectbox("Away", teams, index=1)

if st.button("Predict", type="primary"):
    if home == away:
        st.warning("Pick two different teams.")
    else:
        p = requests.get(f"{API}/predict", params={"home": home, "away": away}, timeout=15).json()["probs"]
        df = pd.DataFrame({"outcome": [f"{home} win", "Draw", f"{away} win"],
                           "p": [p["home_win"], p["draw"], p["away_win"]]})
        st.altair_chart(alt.Chart(df).mark_bar(color="#2b6cb0").encode(
            x=alt.X("p:Q", axis=alt.Axis(format="%"), title="probability"),
            y=alt.Y("outcome:N", sort="-x", title=None)).properties(height=150),
            use_container_width=True)

st.divider()
st.subheader("Championship odds")
top = st.slider("Show top", 5, 32, 16)
odf = pd.DataFrame(get("/odds", top=top)["odds"])
st.altair_chart(alt.Chart(odf).mark_bar(color="#2b6cb0").encode(
    x=alt.X("prob:Q", axis=alt.Axis(format="%"), title="title probability"),
    y=alt.Y("team:N", sort="-x", title=None)).properties(height=22 * len(odf)),
    use_container_width=True)
