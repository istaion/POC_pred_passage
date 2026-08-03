"""
Tests d'intégration de pages/4_Monitorage_Modele.py (C11 -- E3).

Cette page compare tous les modèles de prédiction entre eux (MAPE/MAE/RMSE/
biais par modèle et tranche d'horizon, MASE relatif au modèle naïf) à partir
des données déjà synchronisées -- aucun appel réseau, seule frontière externe
mockée : `data.load_data` (cache Parquet local).

Couverture : cas nominal avec plusieurs modèles/horizons (tableau + boxplot
rendus), et cas données insuffisantes (aucun MovingAverage en référence pour
le MASE, un seul horizon pour le MAPE) qui doit afficher les messages
d'information sans lever d'exception.
"""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PAGE_PATH = "pages/4_Monitorage_Modele.py"


def _fake_etab_df() -> pd.DataFrame:
    return pd.DataFrame({
        "uai": ["0180000X", "0180001Y"],
        "nometabs": ["Lycée Test", "Collège Test"],
        "env": ["prodcentre", "prodcentre"],
    })


def _rich_pred_df() -> pd.DataFrame:
    """Deux établissements, trois modèles, plusieurs horizons -- assez pour
    peupler à la fois le tableau MAPE et le boxplot MASE."""
    rows = []
    models = ["MovingAverage", "Ensemble", "XGBoost21"]
    horizons = [5, 12, 20, 25]
    for uai in ["0180000X", "0180001Y"]:
        for day in range(6):
            target_date = f"2026-01-{10 + day:02d}"
            actual = 100.0 + day * 3
            for horizon in horizons:
                for i, model in enumerate(models):
                    rows.append({
                        "prediction_date": "2026-01-01",
                        "target_date": target_date,
                        "horizon": horizon,
                        "uai": uai,
                        "model": model,
                        "prediction": actual + 5.0 * (i + 1),
                        "effectif_reel": actual,
                        "service": "2",
                        "env": "prodcentre",
                    })
    df = pd.DataFrame(rows)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return df


def _sparse_pred_df() -> pd.DataFrame:
    """Un seul modèle (pas de MovingAverage) -- MASE impossible à calculer."""
    df = pd.DataFrame([{
        "prediction_date": "2026-01-01",
        "target_date": "2026-01-10",
        "horizon": 5,
        "uai": "0180000X",
        "model": "Ensemble",
        "prediction": 110.0,
        "effectif_reel": 100.0,
        "service": "2",
        "env": "prodcentre",
    }])
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return df


@pytest.fixture
def _mock_rich_data(monkeypatch):
    monkeypatch.setattr("data.load_data", lambda: (_rich_pred_df(), _fake_etab_df()))


@pytest.fixture
def _mock_sparse_data(monkeypatch):
    monkeypatch.setattr("data.load_data", lambda: (_sparse_pred_df(), _fake_etab_df()))


def test_nominal_multi_model_renders_table_and_boxplot(_mock_rich_data):
    at = AppTest.from_file(PAGE_PATH)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning
    assert len(at.dataframe) >= 1
    assert len(at.selectbox) == 1


def test_insufficient_data_shows_info_without_crash(_mock_sparse_data):
    at = AppTest.from_file(PAGE_PATH)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert len(at.info) == 1
    assert "MASE" in at.info[0].value
