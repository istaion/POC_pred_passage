"""
Page 4 — Monitorage du modèle : comparaison inter-modèles (C11).

Contrairement à la page 1 (Tableau de bord), centrée sur le modèle envoyé en
production (Ensemble) comparé au modèle naïf, cette page compare tous les
modèles individuels entre eux (ARIMA, Prophet21/35, XGBoost21/35, Ensemble,
GlobalDepXGB...) par tranche d'horizon -- utile pour décider s'il faut
ajuster les poids de l'ensemble ou écarter un modèle qui se dégrade.

Les métriques sont calculées à la volée depuis les données déjà synchronisées
(cache Parquet, cf. sync_data()) -- aucune nouvelle mécanique de cache,
seule la requête de synchronisation a été élargie pour inclure tous les
modèles (auparavant limitée à Ensemble/MovingAverage).
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import (
    HORIZON_BINS,
    build_binned,
    get_mase_by_model,
    get_metrics_by_model,
    load_data,
)

st.title("Monitorage du modèle — Comparaison inter-modèles")
st.caption(
    "Métriques calculées depuis les données synchronisées (passage_predict) — "
    "MAE/MAPE par modèle et tranche d'horizon, MASE relatif au modèle naïf (Moyenne Mobile)."
)

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
    available_envs = sorted(df_pred["env"].dropna().unique().tolist())
    selected_envs = st.multiselect(
        "Environnement", options=available_envs, default=available_envs, placeholder="Tous…"
    )
    if not selected_envs:
        selected_envs = available_envs

df_pred_env = df_pred[df_pred["env"].isin(selected_envs)]
df_binned = build_binned(df_pred_env)

if df_binned.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

models_present = sorted(df_binned["model"].dropna().unique())
st.caption(f"Modèles présents dans les données : {', '.join(models_present)}")

# ---------------------------------------------------------------------------
# Section 1 — MAPE par modèle × horizon
# ---------------------------------------------------------------------------
st.subheader("MAPE (%) par modèle et tranche d'horizon")

df_metrics = get_metrics_by_model(df_binned)

if df_metrics.empty:
    st.info("Pas assez de données pour calculer les métriques par modèle.")
else:
    pivot_mape = df_metrics.pivot(index="Modèle", columns="Horizon", values="MAPE (%)")
    pivot_mape = pivot_mape[[b for b in HORIZON_BINS if b in pivot_mape.columns]]
    st.dataframe(
        pivot_mape.style.format("{:.1f}", na_rep="—").background_gradient(
            cmap="RdYlGn_r", axis=None
        ),
        use_container_width=True,
    )

    with st.expander("Détail complet (MAE, RMSE, biais, N)"):
        st.dataframe(
            df_metrics.sort_values(["Horizon", "MAPE (%)"]),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Section 2 — MASE par modèle (boxplot), référence Moyenne Mobile
# ---------------------------------------------------------------------------
st.subheader("MASE par modèle")
st.caption(
    "MASE = MAE(modèle) / MAE(Moyenne Mobile), calculé établissement par établissement · "
    "< 1 = meilleur que le modèle naïf · ligne rouge = MASE = 1"
)

df_mase = get_mase_by_model(df_binned)

if df_mase.empty:
    st.info(
        "Pas assez de données pour calculer le MASE par modèle "
        "(nécessite le modèle MovingAverage comme référence)."
    )
else:
    selected_bin = st.selectbox("Tranche d'horizon", options=HORIZON_BINS)
    df_mase_bin = df_mase[df_mase["horizon_bin"] == selected_bin]
    models_for_bin = sorted(df_mase_bin["model"].unique())

    if not models_for_bin:
        st.info("Aucun modèle avec assez de données pour cet horizon.")
    else:
        data_for_box = [
            df_mase_bin.loc[df_mase_bin["model"] == m, "mase"].tolist() for m in models_for_bin
        ]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        bp = ax.boxplot(
            data_for_box,
            tick_labels=models_for_bin,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1")
        ax.axhline(y=1, color="red", linestyle="--", linewidth=1.2, label="MASE = 1 (naïf)")
        ax.set_ylabel("MASE", fontsize=10)
        ax.set_title(f"MASE par modèle — horizon {selected_bin}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
