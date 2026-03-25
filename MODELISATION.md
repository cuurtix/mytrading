# Modélisation approfondie du moteur XAUUSD

## Ingestion robuste
- Header detection multi-lignes (Excel mal formaté).
- Mapping fuzzy des colonnes avec alias.
- Parsing timestamp avec détection explicite `epoch_s / epoch_ms / text_datetime`.
- Sanitation OHLC stricte.
- Détection timeframe.
- Merge multi-source par cohérence de timeframe avec logs d’exclusion.

## Structure de marché
- Swings hiérarchiques (mineur/majeur).
- Séquences HH/HL/LH/LL.
- BOS/CHoCH fondés sur swings majeurs.
- Features impulse/retracement/compression/expansion.

## Carte de liquidité
Zones persistantes avec cycle de vie:
- id, bornes, type, side, strength,
- touch_count, age, active,
- swept / last_swept_index.

Types: equal highs/lows, swing highs/lows, range highs/lows, etc.

## FVG lifecycle
Objets FVG persistants:
- création bull/bear,
- mitigation partielle,
- fill complet,
- fill_ratio,
- distance au FVG ouvert le plus proche,
- nombre de FVG ouverts à proximité.

## États contextuels
États:
- RANGE, TREND_UP, TREND_DOWN,
- EXPANSION_UP, EXPANSION_DOWN,
- POST_SWEEP_REVERSAL,
- BREAKOUT_ACCEPTED, BREAKOUT_REJECTED,
- REBALANCING_TO_FVG,
- HIGH_VOLATILITY_PANIC,
- LOW_VOLATILITY_COMPRESSION.

Attribution basée sur structure récente + sweep/FVG + vol + breakout + session.

## Transitions conditionnelles
Apprentissage de:

`P(S_{t+1}|S_t, vol_bucket, liquidity_bucket, breakout_ctx, sweep_ctx, session)`

Ce n’est plus seulement `P(S_{t+1}|S_t)`.

## Distributions conditionnelles
Apprentissage explicite des distributions:
- returns conditionnels (avec continuation 1/3/5 bougies),
- ranges conditionnels,
- mèches conditionnelles,
- continuation/réversal après sweep buy/sell,
- stats FVG (fill complet/partiel, âge de fill),
- profils de session.

## Simulation
À chaque pas:
1. évaluer contexte courant,
2. échantillonner état suivant conditionnel,
3. séparer direction/amplitude,
4. générer OHLC cohérent,
5. mettre à jour liquidité + FVG,
6. conserver le timeframe source.

FOMO est implémenté comme amplification contextuelle bornée (breakout/expansion), jamais comme terme additif magique constant.
