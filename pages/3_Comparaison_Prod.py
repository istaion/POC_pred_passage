"""
Page 3 — Comparaison des prédictions production (API DataBridge) vs effectifs réels (Trino).
"""
import os
import warnings
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from data import (
    ENV_CONFIG,
    SERVICE_CODES,
    compute_bias,
    compute_mae,
    compute_mape,
    compute_rmse,
    fetch_reels_prod,
    load_data,
)

load_dotenv()
warnings.filterwarnings("ignore")

st.title("Comparaison Prédictions Prod vs Réels")

# ---------------------------------------------------------------------------
# Chargement de la liste des établissements (depuis le cache local)
# ---------------------------------------------------------------------------
try:
    _, df_etab = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Token DataBridge (mis en cache dans la session)
# ---------------------------------------------------------------------------
def _gateway_basic_auth() -> tuple[str, str] | None:
    """Basic Auth de la passerelle placée devant l'API en production (toutes
    les routes en sont protégées, en plus du token applicatif /auth/token)."""
    user = os.getenv("IANORD_USERNAME")
    password = os.getenv("IANORD_PASSWORD")
    if user and password:
        return (user, password)
    return None


def _get_api_token() -> str:
    if "databridge_token" not in st.session_state:
        url = f"{os.getenv('DATABRIDGE_URL')}/auth/token"
        resp = requests.post(
            url,
            json={
                "client_key": os.getenv("CLIENT_KEY"),
                "client_secret": os.getenv("CLIENT_SECRET"),
            },
            auth=_gateway_basic_auth(),
        )
        resp.raise_for_status()
        token = resp.json().get("token") or resp.json().get("access_token")
        if not token:
            raise ValueError(f"Réponse /auth/token inattendue : {resp.json()}")
        st.session_state["databridge_token"] = token
    return st.session_state["databridge_token"]


def _reset_token():
    st.session_state.pop("databridge_token", None)


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
col_env, col_etab, col_service = st.columns([1, 3, 1])

with col_env:
    selected_env = st.selectbox("Environnement", options=list(ENV_CONFIG.keys()))

with col_etab:
    df_etab_env = (
        df_etab[df_etab["env"] == selected_env]
        .dropna(subset=["nometabs"])
        .sort_values("nometabs")
    )
    if df_etab_env.empty:
        st.warning("Aucun établissement pour cet environnement. Synchronisez les données.")
        st.stop()
    selected_name = st.selectbox("Établissement", options=df_etab_env["nometabs"].tolist())
    selected_uai = df_etab_env.loc[
        df_etab_env["nometabs"] == selected_name, "uai"
    ].iloc[0]

with col_service:
    selected_service = st.selectbox("Service", options=list(SERVICE_CODES.keys()))

today = date.today()
date_range = st.date_input(
    "Plage de dates",
    value=(today - timedelta(days=30), today),
    format="DD/MM/YYYY",
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range[0], date_range[1]
else:
    start_date, end_date = today - timedelta(days=30), today

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
col_btn, col_reset = st.columns([3, 1])
with col_btn:
    load_clicked = st.button("Charger les données", type="primary", use_container_width=True)
with col_reset:
    if st.button("Réinitialiser le token", use_container_width=True):
        _reset_token()
        st.toast("Token réinitialisé.")

if not load_clicked:
    st.stop()

# Prédictions API
with st.spinner("Récupération des prédictions API…"):
    try:
        token = _get_api_token()
        api_url = (
            f"{os.getenv('DATABRIDGE_URL')}/predict/results/{selected_uai}"
            f"?min_date={start_date}&max_date={end_date}"
            f"&include_all_models=true"
        )
        resp = requests.get(api_url, headers={"X-Api-Token": token}, auth=_gateway_basic_auth())
        if resp.status_code == 401:
            _reset_token()
            st.error("Token expiré — cliquez sur 'Réinitialiser le token' puis rechargez.")
            st.stop()
        resp.raise_for_status()
        records = resp.json()
        if not records:
            st.warning("L'API n'a retourné aucune prédiction pour cette sélection.")
            st.stop()
        raw = pd.DataFrame(records)
        raw = raw[raw["service"] == selected_service].copy()
        raw["date"] = pd.to_datetime(raw["date"])
        # "prediction" = ce qui est envoyé en prod (Ensemble si dispo, sinon XGB…)
        # "models" = détail par modèle individuel
        dates = raw[["date"]].reset_index(drop=True)
        prod_col = raw[["prediction"]].rename(columns={"prediction": "→ Prod"}).reset_index(drop=True)
        if "models" in raw.columns:
            models_df = pd.json_normalize(raw["models"].tolist()).reset_index(drop=True)
        else:
            models_df = pd.DataFrame()
        df_pred_api = pd.concat([dates, prod_col, models_df], axis=1)
        df_pred_api = df_pred_api.sort_values("date").reset_index(drop=True)
    except Exception as e:
        st.error(f"Erreur API : {e}")
        st.stop()

# Effectifs réels Trino
with st.spinner("Récupération des effectifs réels…"):
    try:
        df_reel = fetch_reels_prod(selected_env, selected_uai, selected_service, start_date, end_date)
    except Exception as e:
        st.error(f"Erreur Trino : {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Merge et affichage
# ---------------------------------------------------------------------------
df = (
    df_pred_api.merge(df_reel, on="date", how="outer")
    .sort_values("date")
    .reset_index(drop=True)
)
df["effectif_reel"] = df["effectif_reel"].where(df["effectif_reel"] > 0)

if df.empty:
    st.warning("Aucune donnée à afficher pour cette sélection.")
    st.stop()

model_cols = [c for c in df.columns if c not in ("date", "effectif_reel")]

# ---------------------------------------------------------------------------
# Graphique (prédiction envoyée en prod + réel ; modèles individuels en arrière-plan)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

other_models = [c for c in model_cols if c != "→ Prod"]
for m in other_models:
    ax.plot(df["date"], df[m], color="#AAAAAA", linewidth=0.8, alpha=0.5, label=m)

ax.plot(
    df["date"], df["→ Prod"],
    color="#FF7F0E", linewidth=2.5, marker="o", markersize=4, label="→ Prod (envoyé)",
    zorder=5,
)
ax.plot(
    df["date"], df["effectif_reel"],
    color="#1F77B4", linewidth=2.5, marker="s", markersize=4, label="Effectif réel",
    zorder=5,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax.set_title(
    f"{selected_name} ({selected_uai}) — {selected_service}",
    fontsize=12, fontweight="bold",
)
ax.set_ylabel("Effectif", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Métriques par modèle
# ---------------------------------------------------------------------------
df_both = df.dropna(subset=["effectif_reel"])
df_both = df_both[df_both["effectif_reel"] > 0].reset_index(drop=True)

if not df_both.empty:
    n_days = len(df_both)
    st.subheader(f"Métriques par modèle ({n_days} jours avec effectif réel)")

    metric_rows = []
    for model in model_cols:
        sub = df_both.dropna(subset=[model]).reset_index(drop=True)
        if sub.empty:
            continue
        metric_rows.append({
            "Modèle": model,
            "MAPE (%)": round(compute_mape(sub["effectif_reel"], sub[model]), 2),
            "MAE": round(compute_mae(sub["effectif_reel"], sub[model]), 1),
            "RMSE": round(compute_rmse(sub["effectif_reel"], sub[model]), 1),
            "Biais moyen": round(compute_bias(sub["effectif_reel"], sub[model]), 1),
            "N jours": len(sub),
        })

    if metric_rows:
        df_metrics = pd.DataFrame(metric_rows).set_index("Modèle").sort_values("MAPE (%)")
        st.dataframe(
            df_metrics.style.format("{:.2f}", subset=["MAPE (%)", "MAE", "RMSE", "Biais moyen"])
            .background_gradient(subset=["MAPE (%)"], cmap="RdYlGn_r", axis=0),
            use_container_width=True,
        )
else:
    st.info("Pas d'effectifs réels sur la période — métriques indisponibles.")

# ---------------------------------------------------------------------------
# Tableau détail jour par jour
# ---------------------------------------------------------------------------
st.subheader("Détail jour par jour")
df_display = df.copy()
df_display["date"] = df_display["date"].dt.strftime("%a %d/%m/%Y")
for col in model_cols:
    df_display[col] = df_display[col].round(1)
df_display["effectif_reel"] = df_display["effectif_reel"].round(0)
df_display = df_display.rename(columns={"date": "Date", "effectif_reel": "Réel"})
st.dataframe(df_display.set_index("Date"), use_container_width=True)
