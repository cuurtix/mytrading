# Explication des modèles (XAUUSD uniquement)

## 1) Principe coeur
Le prix est piloté par le flux d'ordres et la liquidité:

- OFI = BuyVolume - SellVolume
- interaction avec poches de liquidité
- impact des ordres de taille élevée
- bruit stochastique

Équation implémentée:

`P(t+1) = P(t) + α*OF_total + β*LiquidityForce + γ*Impact + δ*FOMO + ε`

## 2) Liquidité
On maintient des zones (equal highs/lows, clusters de stops). Chaque zone a:

- un niveau de prix `L_i`
- un poids `w_i`
- une décroissance temporelle `w_i <- decay * w_i`

Force d'attraction:

`LiquidityForce = Σ direction_i * w_i / (|p-L_i|+ε)`

## 3) Sweep
Probabilité:

`P_sweep = sigmoid(k1*LiquidityDensity + k2*Volatility + k3*Momentum)`

Si sweep déclenché:

- accélération vers une zone de liquidité
- continuation (piège breakout) ou reversal (chasse de stops)

## 4) FVG / Imbalance
Détection FVG:

- bullish: `Low[n] > High[n-2]`
- bearish: `High[n] < Low[n-2]`

Intensité imbalance:

`|Body| / Range`

Force de rebalance:

`λ * distance_to_FVG`

## 5) Offre / Demande
Zones représentées par la force des pools de liquidité et décroissance temporelle.
Probabilité de réaction:

`P_reversal = sigmoid(a*ZoneStrength + b*Liquidity + c*Volatility)`

## 6) Structure de marché
Mesures:

- EMA rapide/lente
- signe(EMA_fast - EMA_slow)
- états: RANGE, TREND_UP, TREND_DOWN, EXPANSION, CONSOLIDATION

## 7) FOMO

`FOMO = sigmoid(a1*Momentum + a2*DistanceFromRange + a3*Breakout + a4*LiquidityProximity)`

Puis:

- `OF_FOMO = FOMO * HerdFactor`
- `OF_total = OF_base + OF_FOMO`

Effet: extension de mouvement, overshoot, bulles de court terme.

## 8) Herding
Persistances de flux:

`OF(t) = ρ*OF(t-1) + bruit`

avec `ρ > 0` ⇒ clustering.

## 9) Impact de marché
Loi racine carrée:

`Impact = σ * sqrt(Q/V)`

## 10) Volatilité
Vol de base calibrée par `std(log returns)` puis modulée:

- régime LOW/NORMAL/HIGH/PANIC
- clustering par persistance OF et multiplicateurs de session

## 11) Sessions
Profils:

- Asie: vol basse, liquidité haute
- Londres: activité intermédiaire-haute
- New York: vol/fomo/sweeps amplifiés

Timezone configurable.

## 12) Trading, marge, levier
PnL:

- Long: `(exit-entry)*size`
- Short: `(entry-exit)*size`

Marge:

- position value = `entry*size`
- margin required = `position_value/leverage`

Comptes:

- balance
- equity
- margin
- free margin

## 13) Spread, slippage, frais
- spread croît avec volatilité et baisse de liquidité
- slippage croît avec taille d'ordre / faible liquidité
- frais proportionnels à la valeur notionnelle

## 14) Liquidation
Si `equity < liquidation_ratio * margin_used`, fermeture forcée des positions.
