"""
Tests d'intégration de pages/3_Comparaison_Prod.py (C10 -- E3).

Cette page est la seule du POC à réellement intégrer l'API DataBridge
(POST /auth/token puis GET /predict/results/{uai}) -- cf. rapport_E3.md.

Stratégie : `streamlit.testing.v1.AppTest` exécute le script de la page pour
de vrai (widgets, logique, rendu) ; seules les frontières externes sont
mockées :
  - `requests.post` / `requests.get` (appels HTTP vers l'API DataBridge),
  - `data.load_data` / `data.fetch_reels_prod` (cache Parquet local /
    accès Trino réel pour les effectifs "vrais"), pour ne dépendre ni d'un
    cache local ni d'un vrai accès Trino.

Couverture demandée : succès nominal (200 avec données), token expiré (401),
liste de prédictions vide.
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PAGE_PATH = "pages/3_Comparaison_Prod.py"


def _fake_etab_df() -> pd.DataFrame:
    return pd.DataFrame({
        "uai": ["0180000X"],
        "nometabs": ["Lycée Test"],
        "env": ["prodcentre"],  # doit correspondre à la 1ère clé de ENV_CONFIG
    })


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DATABRIDGE_URL", "https://databridge.test")
    monkeypatch.setenv("CLIENT_KEY", "test-key")
    monkeypatch.setenv("CLIENT_SECRET", "test-secret")


@pytest.fixture(autouse=True)
def _mock_local_data(monkeypatch):
    """Évite toute dépendance au cache Parquet local ou à un vrai accès Trino."""
    monkeypatch.setattr("data.load_data", lambda: (pd.DataFrame(), _fake_etab_df()))
    monkeypatch.setattr(
        "data.fetch_reels_prod",
        lambda *a, **k: pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "effectif_reel": pd.Series(dtype="float64"),
        }),
    )


def _click_load_button(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button[0].click().run()
    return at


def _mock_token_response() -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"access_token": "fake-token"}
    return resp


def test_nominal_200_with_predictions():
    fake_records = [
        {"date": "2026-01-05", "service": "2", "prediction": 120.0,
         "models": {"Ensemble": 120.0, "MovingAverage": 100.0}},
        {"date": "2026-01-06", "service": "2", "prediction": 115.0,
         "models": {"Ensemble": 115.0, "MovingAverage": 98.0}},
    ]
    predict_resp = MagicMock(status_code=200)
    predict_resp.json.return_value = fake_records
    predict_resp.raise_for_status = lambda: None

    with patch("requests.post", return_value=_mock_token_response()), \
         patch("requests.get", return_value=predict_resp):
        at = AppTest.from_file(PAGE_PATH)
        _click_load_button(at)

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error


def test_expired_token_returns_401():
    predict_resp = MagicMock(status_code=401)

    with patch("requests.post", return_value=_mock_token_response()), \
         patch("requests.get", return_value=predict_resp):
        at = AppTest.from_file(PAGE_PATH)
        _click_load_button(at)

    assert not at.exception, [e.value for e in at.exception]
    assert len(at.error) == 1
    assert "expiré" in at.error[0].value
    assert "databridge_token" not in at.session_state


def test_empty_predictions_list():
    predict_resp = MagicMock(status_code=200)
    predict_resp.json.return_value = []
    predict_resp.raise_for_status = lambda: None

    with patch("requests.post", return_value=_mock_token_response()), \
         patch("requests.get", return_value=predict_resp):
        at = AppTest.from_file(PAGE_PATH)
        _click_load_button(at)

    assert not at.exception, [e.value for e in at.exception]
    assert len(at.warning) == 1
    assert "aucune prédiction" in at.warning[0].value.lower()


def test_gateway_basic_auth_sent_when_configured(monkeypatch):
    """La passerelle placée devant l'API en production exige une Basic Auth sur
    toutes les routes, en plus du token applicatif -- vérifie qu'elle est bien
    transmise sur les deux appels (auth/token et predict/results) quand les
    identifiants sont configurés."""
    monkeypatch.setenv("IANORD_USERNAME", "gateway-user")
    monkeypatch.setenv("IANORD_PASSWORD", "gateway-pass")

    predict_resp = MagicMock(status_code=200)
    predict_resp.json.return_value = []
    predict_resp.raise_for_status = lambda: None

    with patch("requests.post", return_value=_mock_token_response()) as mock_post, \
         patch("requests.get", return_value=predict_resp) as mock_get:
        at = AppTest.from_file(PAGE_PATH)
        _click_load_button(at)

    assert not at.exception, [e.value for e in at.exception]
    assert mock_post.call_args.kwargs["auth"] == ("gateway-user", "gateway-pass")
    assert mock_get.call_args.kwargs["auth"] == ("gateway-user", "gateway-pass")


def test_gateway_basic_auth_absent_when_not_configured():
    """Sans identifiants configurés, aucune Basic Auth n'est envoyée (pas de
    crash, comportement rétrocompatible pour un environnement qui n'en a pas
    besoin)."""
    predict_resp = MagicMock(status_code=200)
    predict_resp.json.return_value = []
    predict_resp.raise_for_status = lambda: None

    with patch("requests.post", return_value=_mock_token_response()) as mock_post, \
         patch("requests.get", return_value=predict_resp) as mock_get:
        at = AppTest.from_file(PAGE_PATH)
        _click_load_button(at)

    assert not at.exception, [e.value for e in at.exception]
    assert mock_post.call_args.kwargs["auth"] is None
    assert mock_get.call_args.kwargs["auth"] is None
