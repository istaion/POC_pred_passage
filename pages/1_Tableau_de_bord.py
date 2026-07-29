"""
Page 1 — Tableau de bord global.

Sections :
1. Filtres (environnement, plage de dates, sélection établissements)
2. 4 graphiques lignes (effectif réel vs prédiction Ensemble) par horizon bin
3. Tableau de métriques globales
4. Heatmap MAPE par établissement × horizon bin
5. Box plots MASE par horizon bin
"""
import warnings
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from data import (
    HORIZON_BINS,
    build_binned,
    get_daily_totals,
    get_global_metrics,
    get_mape_by_etab,
    get_mase_by_etab,
    load_data,
)

warnings.filterwarnings("ignore")

st.title("Tableau de bord — Passages en restauration scolaire")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
try:
    df_pred, df_etab = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

df_pred["target_date"] = pd.to_datetime(df_pred["target_date"])

# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
with st.expander("Filtres", expanded=True):
    col_env, col_dates, col_etab = st.columns([1, 2, 3])

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

    with col_dates:
        df_pred_env = df_pred[df_pred["env"].isin(selected_envs)]
        date_min_data = df_pred_env["target_date"].min().date()
        date_max_data = df_pred_env["target_date"].max().date()
        default_start = max(date_min_data, date(2025, 9, 1))

        date_range = st.date_input(
            "Plage de dates",
            value=(default_start, date_max_data),
            min_value=date_min_data,
            max_value=date_max_data,
            format="DD/MM/YYYY",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        else:
            start_date = pd.Timestamp(date_range[0]) if date_range else pd.Timestamp(default_start)
            end_date = pd.Timestamp(date_max_data)

    with col_etab:
        uais_in_data = df_pred_env["uai"].unique()
        df_etab_avail = (
            df_etab[df_etab["uai"].isin(uais_in_data) & df_etab["env"].isin(selected_envs)]
            .dropna(subset=["nometabs"])
            .sort_values("nometabs")
        )
        all_names = df_etab_avail["nometabs"].tolist()
        selected_names = st.multiselect(
            "Établissements",
            options=all_names,
            default=all_names,
            placeholder="Sélectionner des établissements…",
        )
        if not selected_names:
            selected_uais = uais_in_data
        else:
            selected_uais = df_etab_avail.loc[
                df_etab_avail["nometabs"].isin(selected_names), "uai"
            ].tolist()

# ---------------------------------------------------------------------------
# Application des filtres sur df_binned
# ---------------------------------------------------------------------------
df_binned_full = build_binned(df_pred_env)

df_binned = df_binned_full[
    (df_binned_full["target_date"] >= start_date)
    & (df_binned_full["target_date"] <= end_date)
    & (df_binned_full["uai"].isin(selected_uais))
].copy()

n_etab_sel = len(df_binned["uai"].unique())
st.caption(
    f"Période : {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')} · "
    f"{n_etab_sel} établissement(s) · Modèle : Ensemble"
)

if df_binned.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1 — Courbes journalières par horizon bin
# ---------------------------------------------------------------------------
st.subheader("Effectifs journaliers : réels vs prédits")

daily = get_daily_totals(df_binned)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
axes_flat = axes.flatten()

for ax, bin_label in zip(axes_flat, HORIZON_BINS):
    df_d = daily[bin_label]
    ax.plot(
        df_d["target_date"],
        df_d["prediction"],
        color="#FF7F0E",
        linewidth=1.2,
        label="Prédiction Ensemble",
        alpha=0.9,
    )
    ax.plot(
        df_d["target_date"],
        df_d["effectif_reel"],
        color="#1F77B4",
        linewidth=1.2,
        label="Effectif réel",
        alpha=0.9,
    )
    ax2 = ax.twinx()
    ax2.fill_between(
        df_d["target_date"],
        df_d["n_etab"],
        alpha=0.08,
        color="#2CA02C",
        label="N établissements",
    )
    ax2.plot(df_d["target_date"], df_d["n_etab"], color="#2CA02C", linewidth=0.7, alpha=0.5)
    ax2.set_ylabel("N étab.", fontsize=7, color="#2CA02C")
    ax2.tick_params(axis="y", labelsize=7, labelcolor="#2CA02C")
    ax2.set_ylim(0, df_d["n_etab"].max() * 3 if not df_d.empty else 1)
    ax.set_title(f"Horizon {bin_label}", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Effectif total", fontsize=8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle(
    "Somme journalière des effectifs (population identique : étab. avec effectif réel)\n"
    "Courbe verte = nombre d'établissements contribuant",
    fontsize=11,
    fontweight="bold",
)
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Section 2 — Métriques globales
# ---------------------------------------------------------------------------
st.subheader("Métriques globales")

def _metrics_table(df_b: pd.DataFrame):
    df_m = get_global_metrics(df_b)
    st.dataframe(
        df_m.style.format("{:.2f}").background_gradient(
            subset=["MAPE (%)"], cmap="RdYlGn_r", axis=0
        ),
        use_container_width=True,
    )

uais_colleges = set(
    df_etab.loc[df_etab["nometabs"].str.startswith("CL", na=False), "uai"]
)
uais_lycees = set(
    df_etab.loc[df_etab["nometabs"].str.startswith("L", na=False), "uai"]
)

tab_global, tab_colleges, tab_lycees = st.tabs(["Tous", "Collèges (CL…)", "Lycées (L…)"])

with tab_global:
    _metrics_table(df_binned)

with tab_colleges:
    df_binned_cl = df_binned[df_binned["uai"].isin(uais_colleges)]
    if df_binned_cl.empty:
        st.info("Aucun collège dans la sélection courante.")
    else:
        n_cl = df_binned_cl["uai"].nunique()
        st.caption(f"{n_cl} collège(s) dont le nom commence par « CL »")
        _metrics_table(df_binned_cl)

with tab_lycees:
    df_binned_ly = df_binned[df_binned["uai"].isin(uais_lycees)]
    if df_binned_ly.empty:
        st.info("Aucun lycée dans la sélection courante.")
    else:
        n_ly = df_binned_ly["uai"].nunique()
        st.caption(f"{n_ly} lycée(s) dont le nom commence par « L »")
        _metrics_table(df_binned_ly)

# ---------------------------------------------------------------------------
# Section 3 — Heatmap MAPE par établissement × horizon bin
# ---------------------------------------------------------------------------
st.subheader("Heatmap MAPE (%) par établissement")

df_etab_sel = df_etab[
    df_etab["uai"].isin(selected_uais) & df_etab["env"].isin(selected_envs)
]
df_heatmap = get_mape_by_etab(df_binned, df_etab_sel)

if df_heatmap.empty:
    st.info("Pas assez de données pour afficher la heatmap.")
else:
    n_etab = len(df_heatmap)
    fig_h, ax_h = plt.subplots(figsize=(8, max(4, n_etab * 0.35)), constrained_layout=True)

    sns.heatmap(
        df_heatmap,
        ax=ax_h,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "MAPE (%)"},
        annot_kws={"size": 7},
    )
    ax_h.set_xlabel("Horizon", fontsize=9)
    ax_h.set_ylabel("")
    ax_h.tick_params(axis="y", labelsize=7)
    ax_h.tick_params(axis="x", labelsize=9)
    st.pyplot(fig_h)
    plt.close(fig_h)

# ---------------------------------------------------------------------------
# Section 4 — Box plots MASE par horizon bin
# ---------------------------------------------------------------------------
st.subheader("Distribution du MASE par horizon bin")
st.caption(
    "MASE = MAE(Ensemble) / MAE(Moyenne Mobile) · Ligne rouge = MASE 1 (équivalent au modèle naïf)"
)

mase_data = get_mase_by_etab(df_binned)

data_for_box = [mase_data[b] for b in HORIZON_BINS]
non_empty = [d for d in data_for_box if len(d) > 0]

if not non_empty:
    st.info("Pas assez de données pour afficher les box plots MASE.")
else:
    fig_b, ax_b = plt.subplots(figsize=(9, 5), constrained_layout=True)

    bp = ax_b.boxplot(
        data_for_box,
        labels=HORIZON_BINS,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    colors = ["#AED6F1", "#A9DFBF", "#FAD7A0", "#F1948A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax_b.axhline(y=1, color="red", linestyle="--", linewidth=1.2, label="MASE = 1 (naïf)")
    ax_b.set_xlabel("Horizon bin", fontsize=10)
    ax_b.set_ylabel("MASE", fontsize=10)
    ax_b.set_title("MASE par établissement — modèle naïf : Moyenne Mobile", fontsize=11)
    ax_b.legend(fontsize=9)
    ax_b.grid(axis="y", linestyle="--", alpha=0.4)

    st.pyplot(fig_b)
    plt.close(fig_b)
