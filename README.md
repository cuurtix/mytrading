# Moteur local de simulation synthétique XAUUSD (data-driven, structurel)

Ce projet est un moteur **XAUUSD uniquement** qui apprend les comportements depuis vos données (csv/xlsx/xls/zip), puis génère un marché synthétique cohérent **sans replay bougie par bougie**.

## 1) Ingestion des données

Le pipeline (`src/data_ingestion.py`) supporte :
- scan récursif de dossier,
- lecture `csv`, `xlsx`, `xls`,
- lecture `zip` contenant des fichiers dans des sous-dossiers,
- lecture de feuilles Excel,
- détection robuste d’en-tête,
- mapping alias robuste (`timestamp/time/...`, `open/o`, `high/h`, `low/l`, `close/c`, volumes),
- parsing timestamp (epoch sec/ms ou texte),
- validation/sanitation OHLC,
- détection automatique du timeframe,
- logs explicites,
- fusion multi-sources avec exclusion des timeframes incompatibles.

## 2) Détection structurelle

Le moteur extrait des features structurelles (`src/feature_engineering.py`, `src/market_structure.py`) :
- swings mineurs/majeurs,
- séquences HH/HL/LH/LL,
- BOS / CHoCH,
- impulsion / retracement,
- compression / expansion,
- breakout accepté/rejeté,
- position dans le range.

## 3) Carte de liquidité persistante

`src/liquidity_map.py` maintient des zones persistantes avec :
- type, côté, force, âge, touches,
- statut actif/swept,
- distance aux liquidités buy/sell,
- détection sweep (overshoot + réintégration).

## 4) FVG avec cycle de vie

`src/fvg_engine.py` gère des objets FVG persistants :
- création bull/bear,
- mitigation partielle,
- fill complet,
- fill_ratio,
- âge, activité et statistiques de fill.

## 5) États et transitions contextuelles

`src/state_engine.py` apprend des transitions **contextuelles** :

`P(S_{t+1} | S_t, X_t)`

avec contexte (vol bucket, proximité liquidité, breakout/sweep, session, etc.).

## 6) Apprentissage comportemental

`src/behavior_learning.py` apprend des distributions conditionnelles de :
- returns,
- ranges,
- mèches,
- continuation/réversal après sweep,
- statistiques FVG,
- profils par session.

## 7) Simulation synthétique

`src/simulator.py` génère chaque bougie par :
1. contexte courant,
2. transition d’état conditionnelle,
3. séparation direction / amplitude,
4. génération range/wicks conditionnelle,
5. mise à jour liquidité/FVG,
6. respect strict du timeframe source.

## 8) Interface Streamlit

`app.py` permet :
- chargement de fichiers / zip / dossier,
- inspection datasets détectés,
- affichage logs explicites,
- apprentissage,
- visualisation des états et stats conditionnelles,
- simulation calibrée.

## 9) Limites assumées

- Modèle explicite et statistique (pas de microstructure tick-level réelle).
- Les transitions sont conditionnelles par buckets (interprétables), pas modèle profond.
- Qualité dépend de la qualité/couverture des données source XAUUSD.

## Lancer

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```
