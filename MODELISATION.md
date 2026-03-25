# Modélisation (version jeu local)

## Interface
- Front principal HTML/CSS/JS (`web/`)
- Backend local Python (`game_server.py`)
- moteur data-driven inchangé dans `src/`

## Boucle marché vivante
À chaque `step`:
1. extraction contexte récent,
2. transition conditionnelle d’état,
3. génération direction + amplitude,
4. génération OHLC cohérente,
5. mise à jour liquidité/FVG,
6. mark-to-market portefeuille.

## Cohérence des sessions
- timezone référence explicite: UTC
- labels unifiés: `ASIA`, `LONDON_OPEN`, `NEW_YORK`, `LATE_SESSION`

## Cohérence des unités
- distances liquidité/FVG normalisées par échelle locale
- comparaisons uniquement relatives avec seuils relatifs

## Impact des ordres joueur
Les ordres influencent:
- impact prix,
- spread,
- slippage,
- poussée impulsive,
- sweep / stop-hunt cascade.

## Carte de liquidité et FVG
- objets persistants mis à jour incrémentalement
- pas de fuite temporelle (query avant update)

## Heuristiques assumées
- bornes de saturation d’impact et FOMO
- coefficients de conversion ordre->impact
- seuils de bucket relatifs
Ces heuristiques restent explicitement assumées pour conserver une jouabilité stable en local.
