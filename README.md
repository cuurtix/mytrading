# XAUUSD Local Trading Game (HTML/CSS/JS + moteur Python data-driven)

Le projet est maintenant un **jeu local jouable** avec interface principale en **HTML/CSS/JavaScript**.

## Architecture choisie (Option B hybride locale)

- **Frontend local**: `web/index.html`, `web/style.css`, `web/app.js`
- **Backend local Python**: `game_server.py`
- **Moteur**: ingestion + apprentissage structurel + transitions contextuelles + simulation incrémentale

Pourquoi cette architecture:
- garde la puissance du moteur Python existant,
- offre une UI jeu fluide en HTML,
- reste simple à lancer en local.

## Gameplay disponible

- BUY / SELL
- CLOSE ALL
- CLOSE 10% / 20% / 50%
- DEPOSIT / WITHDRAW
- RESTART / RESET TOTAL
- boucle marché vivante (STEP/RUN/PAUSE + vitesse)
- portefeuille temps réel (balance/equity/pnl/marge/exposition/positions)

## Moteur conservé

Le cœur data-driven est conservé:
- ingestion recursive csv/xlsx/xls/zip,
- nettoyage/timeframe,
- features structurelles,
- carte de liquidité incrémentale,
- cycle de vie FVG,
- états contextuels,
- transitions conditionnelles,
- séparation direction/amplitude,
- sessions UTC.

## Impact du joueur sur le marché

Les ordres du joueur influencent:
- impact prix,
- spread,
- slippage,
- probabilité d’impulsion,
- interactions sweep/stop-hunt (cascade).

## Lancer le jeu local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python game_server.py
```

Puis ouvrir: `http://127.0.0.1:8000`

## Outil secondaire

`app.py` (Streamlit) peut rester comme outil secondaire de debug/calibration, mais l’interface principale de jeu est HTML.

## Heuristiques assumées

Certaines bornes de gameplay sont conservées (ex: saturation d’impact/FOMO, seuils de bucket relatifs), explicitement documentées comme compromis jouabilité/réalisme.


Note: CLOSE 10% / 20% / 50% applique ce pourcentage sur **chaque position ouverte**.

- liquidation forcée si equity <= maintenance margin

Les endpoints API renvoient un snapshot complet après chaque action (metrics, positions, last_price, state, timestamp, recent_events).
