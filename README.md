# XAUUSD Local Trading Simulator (produit principal) + Streamlit (outil secondaire)

Ce dépôt fournit deux couches distinctes :
- **Produit principal** : simulateur de trading XAUUSD jouable en local (serveur Flask + frontend web live).
- **Outil secondaire** : `app.py` Streamlit pour debug/calibration, non destiné à l’usage “jeu”.

---

## Architecture cible (résumé)

1. `src/` = noyau moteur (ingestion, calibration, comportements, transitions, exécution portefeuille).  
2. `game_server.py` = API locale du jeu (`/api/init`, `/api/step`, `/api/order`, ...).  
3. `web/` = interface de simulation (chart chandeliers, boutons de trading, positions, métriques, events).  
4. `app.py` = utilitaire de calibration (facultatif), volontairement séparé du parcours joueur.

## Architecture choisie

- **Frontend local (principal)** : `web/index.html`, `web/style.css`, `web/app.js`
- **Backend local Python (principal)** : `game_server.py`
- **Moteur** : `src/` (ingestion, calibration, génération synthétique, trading, validation logique)
- **Configuration moteur** : `src/model_config.py` (paramètres impact/liquidité/cascade/FVG)

Pourquoi cette architecture:
- garde la puissance du moteur Python existant,
- offre une UI jeu fluide en HTML,
- reste simple à lancer en local.

## Fonctionnalités de simulation (UI web)

- BUY / SELL
- CLOSE ALL
- CLOSE 10% / 20% / 50%
- DEPOSIT / WITHDRAW
- RESTART / RESET TOTAL
- boucle marché vivante (STEP/RUN/PAUSE + vitesse)
- graphique chandeliers OHLC en temps réel + bouton RECENTER
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

## Lancer le simulateur (commande principale)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python game_server.py
```

Puis ouvrir: `http://127.0.0.1:8000`

## Mode debug/calibration (facultatif)

`app.py` est conservé comme **outil secondaire** :

```bash
streamlit run app.py
```

Ce mode ne remplace pas l’application produit principale.

## Heuristiques assumées

Certaines bornes de gameplay sont conservées (ex: saturation d’impact/FOMO, seuils de bucket relatifs), explicitement documentées comme compromis jouabilité/réalisme.


Note: CLOSE 10% / 20% / 50% applique ce pourcentage sur **chaque position ouverte**.

- liquidation forcée si equity <= maintenance margin

Les endpoints API renvoient un snapshot complet après chaque action (metrics, positions, last_price, state, timestamp, recent_events).

## Détail de la référence macro or (garde-fou)

- Référence configurable: `GLOBAL_GOLD_REFERENCE_DAILY_NOTIONAL = 3.27e11` (USD/jour, ordre de grandeur global).
- Cette référence sert uniquement de **garde-fou macro**.
- Le gameplay est piloté d’abord par la liquidité locale (session + volume local + carte de liquidité), pas par ce volume macro.
