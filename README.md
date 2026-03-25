# XAUUSD Local Trading Game (HTML/CSS/JS + moteur Python data-driven)

Le projet est maintenant un **jeu local jouable** avec interface principale en **HTML/CSS/JavaScript**.

## Architecture choisie (Option B hybride locale)

- **Frontend local**: `web/index.html`, `web/style.css`, `web/app.js`
- **Backend local Python**: `game_server.py`
- **Moteur**: ingestion + apprentissage structurel + transitions contextuelles + simulation incrémentale
- **Configuration moteur**: `src/model_config.py` (centralisation des paramètres impact/liquidité/cascade/FVG)

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

Les ordres du joueur **n’impactent pas systématiquement** le marché.

Le moteur applique une loi d’impact bornée de type square-root, avec:
- seuil d’activation (petits ordres => impact nul/quasi nul),
- participation relative à la **liquidité locale exécutable**,
- garde-fou macro configurable pour éviter des extrêmes non réalistes.

En pratique:
- petits ordres => spread/slippage très faibles, pas de cascade artificielle,
- ordres plus gros vs liquidité locale => impact plus visible,
- impact borné et pression prix résiduelle décroissante pour préserver la stabilité.

La configuration associée est centralisée dans `ModelConfig` pour éviter les règles magiques dispersées.

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

## Détail de la référence macro or (garde-fou)

- Référence configurable: `GLOBAL_GOLD_REFERENCE_DAILY_NOTIONAL = 3.27e11` (USD/jour, ordre de grandeur global).
- Cette référence sert uniquement de **garde-fou macro**.
- Le gameplay est piloté d’abord par la liquidité locale (session + volume local + carte de liquidité), pas par ce volume macro.
