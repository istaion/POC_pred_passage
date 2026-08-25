# POC Passages

Application de démonstration (Streamlit) construite autour du service de prédiction de
fréquentation en restauration scolaire (projet `IAnord Data Bridge`). Elle restitue les prédictions
du modèle, les compare aux effectifs réels observés en production, et permet de suivre la dérive
des métriques d'erreur par modèle.

## Besoin

*« En tant que chef de cuisine, je veux connaître le nombre de passages un mois à l'avance, afin
d'anticiper mes commandes de denrées. »* — c'est le besoin fondateur du projet `prediction_passages`
que cette application restitue et met à l'épreuve.

## Architecture

L'application lit les données de deux façons différentes selon la page :

- **Cache local (Parquet)**, synchronisé à la demande (bouton « Synchroniser ») depuis l'entrepôt
  Trino — utilisé par les pages "Tableau de bord", "Par établissement" et "Monitorage modèle".
- **API IAnord Data Bridge**, interrogée en direct (authentification par token) — utilisée par la
  page "Comparaison Prod", qui superpose la prédiction retournée par l'API aux effectifs réels lus
  directement dans l'entrepôt.

Voir `assets_E4/flux_contexte.png` et `assets_E4/flux_poc.png` (dépôt `IAnord_data_bridge`) pour les
diagrammes de flux de données complets.

## Pages

| Page | Contenu |
|---|---|
| Tableau de bord | Vue d'ensemble des prédictions vs effectifs réels, tous établissements |
| Par établissement | Historique détaillé d'un établissement |
| Comparaison Prod | Intégration réelle à l'API (auth, consultation, gestion de l'expiration du token) |
| Monitorage modèle | MAPE/MASE par modèle et tranche d'horizon, pour suivre la dérive du modèle en production |

## Installation

Nécessite [uv](https://docs.astral.sh/uv/) et Python 3.14.

```bash
uv sync --group dev
cp .env.example .env   # puis renseigner les identifiants (voir ci-dessous)
```

### Variables d'environnement (`.env`)

| Variable | Usage |
|---|---|
| `OVH_API_KEY` / `OVH_SECRET_KEY` | Accès en lecture à l'entrepôt Trino (OVH Data Platform) |
| `DATABRIDGE_URL` | URL de l'API IAnord Data Bridge |
| `CLIENT_KEY` / `CLIENT_SECRET` | Identifiants d'authentification applicative à cette API |
| `IANORD_USERNAME` / `IANORD_PASSWORD` | Basic Auth de la passerelle placée devant l'API en production — protège toutes les routes, en amont du token applicatif ci-dessus |

## Exécution

```bash
uv run streamlit run app.py
```

## Tests

```bash
uv run pytest tests/ -v
```

7 tests d'intégration (`streamlit.testing.v1.AppTest`), sur les pages "Comparaison Prod" (intégration
API, dont la Basic Auth de la passerelle) et "Monitorage modèle" : appels réseau et accès à
l'entrepôt de données simulés, aucune dépendance à un accès réel pour exécuter la suite.

## Déploiement

Déployé sur [Render](https://render.com) à partir de ce dépôt (voir `render.yaml` et `Dockerfile`).
Le déploiement est déclenché par la chaîne d'intégration continue (`.github/workflows/ci.yml`)
uniquement après succès des tests — voir ce fichier pour le détail de la chaîne.
