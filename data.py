"""
Chargement des données depuis Trino avec cache Parquet local.
"""
import os
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from trino.auth import BasicAuthentication
from trino.dbapi import connect

load_dotenv()

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_PREDICTIONS = CACHE_DIR / "predictions.parquet"
CACHE_ETABLISSEMENTS = CACHE_DIR / "etablissements.parquet"

HORIZON_BINS = ["≤7j", "≤14j", "≤21j", ">21j"]
BIN_ORDER = {b: i for i, b in enumerate(HORIZON_BINS)}

# Mettre à jour ici lors des migrations de tables
ENV_CONFIG = {
    "prodcentre": {
        # Transitoire — sera remplacé par db_mg6jk45h_webgerest_centre.webgerest_centre.centr_passage_predict
        "predictions": "db_mg6jk45h_default_dataset.default_dataset.passage_predict",
        "etablissements": "db_mg6jk45h_webgerest_centre.webgerest_centre.centr_login",
        "effect": "db_mg6jk45h_webgerest_centre.webgerest_centre.centr_effect",
        "api_env": "prodcentre",
    },
    "prod13": {
        "predictions": "db_mg6jk45h_prod13.prod13.wg_13_passage_predict",
        "etablissements": "db_mg6jk45h_prod13.prod13.wg_13_login",
        "effect": "db_mg6jk45h_prod13.prod13.wg_13_effect",
        "api_env": "prod13",
    },
}

# Mapping service label → code en base (codss2)
SERVICE_CODES: dict[str, str] = {
    "DEJEUNER": "2",
}


def _get_connection():
    return connect(
        host="@data-ianord-query.eu.dataplatform.ovh.net",
        port=443,
        user=os.getenv("OVH_API_KEY"),
        auth=BasicAuthentication(os.getenv("OVH_API_KEY"), os.getenv("OVH_SECRET_KEY")),
        catalog="db_mg6jk45h_default_dataset",
        schema="default_dataset",
        http_scheme="https",
    )


def _fetch_predictions() -> pd.DataFrame:
    conn = _get_connection()
    frames = []
    for env, cfg in ENV_CONFIG.items():
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT
                prediction_date,
                target_date,
                horizon,
                uai,
                model,
                prediction,
                effectif_reel,
                service
            FROM {cfg["predictions"]}
            WHERE target_date >= DATE '2024-09-01'
        """)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)
        df["env"] = env
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["horizon"] = df["horizon"].astype(int)
    # Certains envs stockent effectif_reel sur un seul modèle : on le propage à tous
    df["effectif_reel"] = df.groupby(
        ["target_date", "uai", "service", "env"]
    )["effectif_reel"].transform("max")
    return df


def _fetch_etablissements() -> pd.DataFrame:
    conn = _get_connection()
    frames = []
    for env, cfg in ENV_CONFIG.items():
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT login AS uai, nometabs, logingroupe
            FROM {cfg["etablissements"]}
        """)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)
        df["env"] = env
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["uai", "env"])


def sync_data() -> None:
    """Requête Trino et écrase les fichiers Parquet de cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    df_pred = _fetch_predictions()
    df_etab = _fetch_etablissements()
    df_pred.to_parquet(CACHE_PREDICTIONS, index=False)
    df_etab.to_parquet(CACHE_ETABLISSEMENTS, index=False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge depuis le cache Parquet ; lance une erreur si absent."""
    if not CACHE_PREDICTIONS.exists() or not CACHE_ETABLISSEMENTS.exists():
        raise FileNotFoundError(
            "Cache absent. Cliquez sur 'Synchroniser les données' dans la barre latérale."
        )
    df_pred = pd.read_parquet(CACHE_PREDICTIONS)
    df_etab = pd.read_parquet(CACHE_ETABLISSEMENTS)
    return df_pred, df_etab


def fetch_reels_prod(
    env: str,
    uai: str,
    service: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Effectifs réels depuis la table effect de l'environnement donné.
    Retourne un DataFrame avec colonnes [date, effectif_reel].
    """
    service_code = SERVICE_CODES[service]
    table = ENV_CONFIG[env]["effect"]
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT efdate AS date, SUM(efreel) AS effectif_reel
        FROM {table}
        WHERE login_site = '{uai}'
          AND codss2 = '{service_code}'
          AND efdate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        GROUP BY efdate
        ORDER BY efdate
    """)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def last_sync_time() -> str | None:
    """Retourne la date/heure de la dernière synchronisation ou None."""
    if CACHE_PREDICTIONS.exists():
        ts = CACHE_PREDICTIONS.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
    return None


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def assign_horizon_bin(horizon: int) -> str:
    if horizon <= 7:
        return "≤7j"
    elif horizon <= 14:
        return "≤14j"
    elif horizon <= 21:
        return "≤21j"
    else:
        return ">21j"


def build_binned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (target_date, uai, service, model, horizon_bin, env), garde
    exactement une ligne — la plus récente (prediction_date max) parmi
    celles qui ont le plus petit horizon dans le bin.
    """
    df = df.copy()
    df = (
        df.sort_values("prediction_date", ascending=False)
        .drop_duplicates(subset=["target_date", "uai", "service", "model", "horizon", "env"])
    )
    df["horizon_bin"] = df["horizon"].apply(assign_horizon_bin)
    df_binned = (
        df.sort_values("horizon")
        .groupby(["target_date", "uai", "service", "model", "horizon_bin", "env"], as_index=False)
        .first()
    )
    return df_binned


def count_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic : retourne les groupes qui ont plus d'une ligne dans le DataFrame brut.
    """
    key = ["target_date", "uai", "service", "model", "horizon", "env"]
    counts = df.groupby(key).size().reset_index(name="n")
    return counts[counts["n"] > 1].sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_mape(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & (actual > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def compute_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & (actual > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def compute_bias(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & (actual > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(predicted[mask] - actual[mask]))


def compute_mase(
    ensemble_preds: pd.Series,
    ma_preds: pd.Series,
    actual: pd.Series,
) -> float:
    mask = actual.notna() & (actual > 0)
    if mask.sum() == 0:
        return np.nan
    mae_ens = np.mean(np.abs(actual[mask] - ensemble_preds[mask]))
    mae_ma = np.mean(np.abs(actual[mask] - ma_preds[mask]))
    if mae_ma == 0:
        return np.nan
    return float(mae_ens / mae_ma)


# ---------------------------------------------------------------------------
# Helpers pour les pages
# ---------------------------------------------------------------------------

def get_global_metrics(df_binned: pd.DataFrame) -> pd.DataFrame:
    """Tableau de métriques globales par horizon bin (modèle Ensemble)."""
    ens = df_binned[df_binned["model"] == "Ensemble"]
    rows = []
    for b in HORIZON_BINS:
        sub = ens[ens["horizon_bin"] == b].dropna(subset=["effectif_reel"])
        rows.append({
            "Horizon": b,
            "MAPE (%)": round(compute_mape(sub["effectif_reel"], sub["prediction"]), 2),
            "MAE": round(compute_mae(sub["effectif_reel"], sub["prediction"]), 1),
            "RMSE": round(compute_rmse(sub["effectif_reel"], sub["prediction"]), 1),
            "Biais moyen": round(compute_bias(sub["effectif_reel"], sub["prediction"]), 1),
        })
    return pd.DataFrame(rows).set_index("Horizon")


def get_mape_by_etab(df_binned: pd.DataFrame, df_etab: pd.DataFrame) -> pd.DataFrame:
    """MAPE par établissement × horizon bin (pour la heatmap)."""
    ens = df_binned[df_binned["model"] == "Ensemble"].dropna(subset=["effectif_reel"])
    ens = ens[ens["effectif_reel"] > 0]
    pivot_rows = []
    for (uai, env), grp in ens.groupby(["uai", "env"]):
        row = {"uai": uai, "env": env}
        for b in HORIZON_BINS:
            sub = grp[grp["horizon_bin"] == b]
            row[b] = compute_mape(sub["effectif_reel"], sub["prediction"])
        pivot_rows.append(row)
    if not pivot_rows:
        return pd.DataFrame(columns=HORIZON_BINS)
    df_pivot = pd.DataFrame(pivot_rows)
    df_pivot = df_pivot.merge(df_etab[["uai", "env", "nometabs"]], on=["uai", "env"], how="left")
    df_pivot["nometabs"] = df_pivot["nometabs"].fillna(df_pivot["uai"])
    df_pivot = df_pivot.set_index("nometabs")[HORIZON_BINS]
    return df_pivot


def get_mase_by_etab(df_binned: pd.DataFrame) -> dict[str, list[float]]:
    """MASE par établissement pour chaque horizon bin (pour les box plots)."""
    result: dict[str, list[float]] = {b: [] for b in HORIZON_BINS}
    for _, sub_uai in df_binned.groupby(["uai", "env"]):
        for b in HORIZON_BINS:
            sub_bin = sub_uai[sub_uai["horizon_bin"] == b]
            ens = sub_bin[sub_bin["model"] == "Ensemble"]
            ma = sub_bin[sub_bin["model"] == "MovingAverage"]
            merged = ens[["target_date", "service", "prediction", "effectif_reel"]].merge(
                ma[["target_date", "service", "prediction"]],
                on=["target_date", "service"],
                suffixes=("_ens", "_ma"),
            )
            merged = merged.dropna(subset=["effectif_reel"])
            mase = compute_mase(merged["prediction_ens"], merged["prediction_ma"], merged["effectif_reel"])
            if not np.isnan(mase):
                result[b].append(mase)
    return result


def get_week_metrics(df_week_daily: pd.DataFrame) -> pd.DataFrame | None:
    """
    Métriques sur la semaine affichée (page 2).
    df_week_daily doit avoir les colonnes effectif_reel et prediction.
    Retourne None si pas assez de données réelles.
    """
    sub = df_week_daily.dropna(subset=["effectif_reel"])
    sub = sub[sub["effectif_reel"] > 0]
    if sub.empty:
        return None
    actual = sub["effectif_reel"]
    pred = sub["prediction"]
    return pd.DataFrame([{
        "MAPE (%)": round(compute_mape(actual, pred), 2),
        "MAE": round(compute_mae(actual, pred), 1),
        "RMSE": round(compute_rmse(actual, pred), 1),
        "Biais moyen": round(compute_bias(actual, pred), 1),
        "N jours": int(len(sub)),
    }], index=["Semaine"])


def get_daily_totals(df_binned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Somme journalière de effectif_reel et prediction (Ensemble) par horizon bin.

    Pour chaque target_date, seuls les (uai, service) ayant effectif_reel > 0
    sont inclus dans les deux sommes.
    """
    ens = df_binned[df_binned["model"] == "Ensemble"]
    result = {}
    for b in HORIZON_BINS:
        sub = ens[ens["horizon_bin"] == b]
        sub_with_real = sub[sub["effectif_reel"] > 0]
        daily = sub_with_real.groupby("target_date", as_index=False).agg(
            effectif_reel=("effectif_reel", "sum"),
            prediction=("prediction", "sum"),
            n_etab=("uai", "nunique"),
        )
        daily = daily.sort_values("target_date")
        result[b] = daily
    return result


# ---------------------------------------------------------------------------
# Monitorage inter-modèles (C11) — au-delà d'Ensemble vs modèle naïf
# ---------------------------------------------------------------------------

def get_metrics_by_model(df_binned: pd.DataFrame) -> pd.DataFrame:
    """Tableau de métriques (MAPE/MAE/RMSE/Biais) par (modèle, horizon bin)."""
    rows = []
    for model in sorted(df_binned["model"].dropna().unique()):
        sub_model = df_binned[df_binned["model"] == model]
        for b in HORIZON_BINS:
            sub = sub_model[sub_model["horizon_bin"] == b].dropna(subset=["effectif_reel"])
            sub = sub[sub["effectif_reel"] > 0]
            if sub.empty:
                continue
            rows.append({
                "Modèle": model,
                "Horizon": b,
                "MAPE (%)": round(compute_mape(sub["effectif_reel"], sub["prediction"]), 2),
                "MAE": round(compute_mae(sub["effectif_reel"], sub["prediction"]), 1),
                "RMSE": round(compute_rmse(sub["effectif_reel"], sub["prediction"]), 1),
                "Biais moyen": round(compute_bias(sub["effectif_reel"], sub["prediction"]), 1),
                "N": int(len(sub)),
            })
    return pd.DataFrame(rows)


def get_mase_by_model(df_binned: pd.DataFrame) -> pd.DataFrame:
    """
    MASE par (modèle, établissement, horizon bin), modèle naïf de référence :
    MovingAverage (cf. page 1). Une ligne par (uai, horizon bin, modèle).
    """
    rows = []
    for (uai, env), grp in df_binned.groupby(["uai", "env"]):
        for b in HORIZON_BINS:
            sub_bin = grp[grp["horizon_bin"] == b]
            ma = sub_bin[sub_bin["model"] == "MovingAverage"]
            if ma.empty:
                continue
            for model in sub_bin["model"].unique():
                if model == "MovingAverage":
                    continue
                mod = sub_bin[sub_bin["model"] == model]
                merged = mod[["target_date", "service", "prediction", "effectif_reel"]].merge(
                    ma[["target_date", "service", "prediction"]],
                    on=["target_date", "service"],
                    suffixes=("_mod", "_ma"),
                )
                merged = merged.dropna(subset=["effectif_reel"])
                merged = merged[merged["effectif_reel"] > 0]
                if merged.empty:
                    continue
                mase = compute_mase(merged["prediction_mod"], merged["prediction_ma"], merged["effectif_reel"])
                if not np.isnan(mase):
                    rows.append({"uai": uai, "env": env, "horizon_bin": b, "model": model, "mase": mase})
    return pd.DataFrame(rows, columns=["uai", "env", "horizon_bin", "model", "mase"])
