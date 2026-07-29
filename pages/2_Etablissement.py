"""
Page 2 — Détail par établissement.

Affiche les effectifs réels et prédits (Ensemble) pour la semaine
lundi–samedi de référence, avec navigation semaine précédente/suivante.
"""
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data import build_binned, get_week_metrics, load_data

warnings.filterwarnings("ignore")

# Borne inférieure de navigation : semaine de septembre 2024
MIN_DATE = pd.Timestamp("2024-09-01")
# On prend le lundi de la semaine contenant le 2024-09-01
_weekday_min = MIN_DATE.weekday()
MIN_MONDAY = MIN_DATE - timedelta(days=_weekday_min)

st.title("Détail par établissement")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
try:
    df_pred, df_etab = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Sélection de l'établissement
# ---------------------------------------------------------------------------
df_etab_sorted = df_etab.dropna(subset=["nometabs"]).sort_values("nometabs")

col_env, col_etab = st.columns([1, 3])

with col_env:
    available_envs = sorted(df_pred["env"].dropna().unique().tolist())
    selected_envs = st.multiselect(
        "Environnement",
        options=available_envs,
        default=available_envs,
        placeholder="Tous…",
    )
    if not selected_envs:
        selected_envs = available_envs

with col_etab:
    uais_with_data = df_pred[df_pred["env"].isin(selected_envs)]["uai"].unique()
    df_etab_filtered = (
        df_etab_sorted[
            df_etab_sorted["uai"].isin(uais_with_data)
            & df_etab_sorted["env"].isin(selected_envs)
        ]
    )
    if df_etab_filtered.empty:
        st.warning("Aucun établissement trouvé pour cet environnement.")
        st.stop()
    options = df_etab_filtered["nometabs"].tolist()
    selected_name = st.selectbox("Établissement", options)

selected_uai = df_etab_filtered.loc[
    df_etab_filtered["nometabs"] == selected_name, "uai"
].iloc[0]

# ---------------------------------------------------------------------------
# Données de l'établissement (modèle Ensemble uniquement)
# ---------------------------------------------------------------------------
df_etab_data = df_pred[
    (df_pred["uai"] == selected_uai)
    & (df_pred["model"] == "Ensemble")
    & (df_pred["env"].isin(selected_envs))
].copy()

if df_etab_data.empty:
    st.info("Aucune donnée disponible pour cet établissement.")
    st.stop()

df_etab_data["target_date"] = pd.to_datetime(df_etab_data["target_date"])

# Pour chaque target_date, garder la ligne avec l'horizon minimal
df_etab_min = (
    df_etab_data.sort_values("horizon")
    .groupby(["target_date", "service"], as_index=False)
    .first()
)

# ---------------------------------------------------------------------------
# Trouver la semaine de référence (dernière semaine avec effectif_reel > 0)
# ---------------------------------------------------------------------------
has_real = df_etab_min[df_etab_min["effectif_reel"] > 0].copy()

if has_real.empty:
    st.info("Aucun effectif réel enregistré pour cet établissement.")
    st.stop()

last_real_date = has_real["target_date"].max()
weekday = last_real_date.weekday()  # 0=lundi
ref_monday = last_real_date - timedelta(days=weekday)

# ---------------------------------------------------------------------------
# Navigation semaine
# ---------------------------------------------------------------------------
if "week_offset" not in st.session_state:
    st.session_state["week_offset"] = 0

# Réinitialiser l'offset quand on change d'établissement
if st.session_state.get("_last_uai") != selected_uai:
    st.session_state["week_offset"] = 0
    st.session_state["_last_uai"] = selected_uai

offset = st.session_state["week_offset"]
current_monday = ref_monday + timedelta(weeks=offset)
current_saturday = current_monday + timedelta(days=5)

# Bornes de navigation
at_min = current_monday <= MIN_MONDAY
at_max = offset >= 0

col_prev, col_title, col_next = st.columns([1, 4, 1])

with col_prev:
    if st.button("← Semaine précédente", use_container_width=True, disabled=at_min):
        st.session_state["week_offset"] -= 1
        st.rerun()

with col_title:
    st.markdown(
        f"<h4 style='text-align:center'>{current_monday.strftime('%d/%m/%Y')} — "
        f"{current_saturday.strftime('%d/%m/%Y')}</h4>",
        unsafe_allow_html=True,
    )

with col_next:
    if st.button("Semaine suivante →", use_container_width=True, disabled=at_max):
        st.session_state["week_offset"] += 1
        st.rerun()

# ---------------------------------------------------------------------------
# Filtrer les données sur la semaine courante
# ---------------------------------------------------------------------------
mask_week = (df_etab_min["target_date"] >= current_monday) & (
    df_etab_min["target_date"] <= current_saturday
)
df_week = df_etab_min[mask_week].copy()

# Agréger les services (somme par jour)
df_week_daily = df_week.groupby("target_date", as_index=False).agg(
    prediction=("prediction", "sum"),
    effectif_reel=("effectif_reel", "sum"),
)
df_week_daily = df_week_daily.sort_values("target_date")

# Masquer effectif_reel = 0 (non renseigné)
df_week_daily.loc[df_week_daily["effectif_reel"] == 0, "effectif_reel"] = np.nan

# ---------------------------------------------------------------------------
# Graphique
# ---------------------------------------------------------------------------
JOURS_FR = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam"}

if df_week_daily.empty:
    st.info("Aucune donnée pour cette semaine.")
    st.stop()

fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

x_labels = [
    f"{JOURS_FR.get(d.weekday(), '')} {d.strftime('%d/%m')}"
    for d in df_week_daily["target_date"]
]
x = range(len(x_labels))

ax.plot(
    x,
    df_week_daily["prediction"].values,
    color="#FF7F0E",
    linewidth=2,
    marker="o",
    markersize=5,
    label="Prédiction Ensemble",
)

real = df_week_daily.dropna(subset=["effectif_reel"])
if not real.empty:
    real_x = [
        list(df_week_daily["target_date"]).index(d)
        for d in real["target_date"]
    ]
    ax.plot(
        real_x,
        real["effectif_reel"].values,
        color="#1F77B4",
        linewidth=2,
        marker="s",
        markersize=5,
        label="Effectif réel",
    )

ax.set_xticks(list(x))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Effectif", fontsize=10)
ax.set_title(
    f"{selected_name} ({selected_uai}) — semaine du {current_monday.strftime('%d/%m/%Y')}",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Métriques de la semaine
# ---------------------------------------------------------------------------
df_metrics_week = get_week_metrics(df_week_daily)

if df_metrics_week is not None:
    st.subheader("Métriques de la semaine")
    cols = st.columns(5)
    labels = ["MAPE (%)", "MAE", "RMSE", "Biais moyen", "N jours"]
    for col, lbl in zip(cols, labels):
        val = df_metrics_week.loc["Semaine", lbl]
        col.metric(lbl, val)
else:
    st.info("Pas d'effectifs réels sur cette semaine — métriques indisponibles.")

# ---------------------------------------------------------------------------
# Mini tableau récapitulatif
# ---------------------------------------------------------------------------
df_display = df_week_daily.rename(
    columns={"target_date": "Date", "prediction": "Prédit", "effectif_reel": "Réel"}
).copy()
df_display["Date"] = df_display["Date"].dt.strftime("%a %d/%m")
df_display["Prédit"] = df_display["Prédit"].round(0).astype(int)
df_display["Réel"] = df_display["Réel"].where(df_display["Réel"].notna(), other="—")
st.dataframe(df_display.set_index("Date"), use_container_width=True)
