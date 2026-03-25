# Simulateur local de marché synthétique XAUUSD

Ce projet construit un **moteur de marché synthétique** (et non un replay/backtest classique) dédié exclusivement à **XAUUSD**.

## Objectif

Générer une évolution de prix réaliste via microstructure:

- déséquilibre de flux d'ordres (OFI)
- attraction vers la liquidité
- impact de marché des gros ordres
- composante comportementale FOMO / herding
- régimes de volatilité et sessions (Asie/Londres/New York)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'interface

```bash
streamlit run app.py
```

## Données

Déposer les CSV dans `data/` au format:

```text
datetime,open,high,low,close,volume
```

## Architecture

- `src/data_learning.py`: calibration statistique depuis historique
- `src/market_models.py`: modèles microstructure/liquidité/FOMO/impact/volatilité
- `src/simulator.py`: moteur de simulation synthétique XAUUSD
- `src/trading.py`: exécution ordres, spread/slippage/frais, PnL, liquidation
- `app.py`: UI locale Streamlit

## Formules clés

### Dynamique de prix

\[
P_{t+1}=P_t + \alpha OF_{total,t} + \beta LiquidityForce_t + \gamma Impact_t + \delta FOMO_t + \varepsilon_t
\]

avec

- \(OF_{total}=OF_{base}+OF_{FOMO}\)
- \(OF_{FOMO}=FOMO\cdot HerdFactor\)
- \(Impact = \sigma \sqrt{Q/V}\)
- \(\varepsilon\sim\mathcal N(0,\sigma^2)\)

### Liquidité

\[
LiquidityForce(p) = \sum_i \frac{w_i}{|p-L_i|+\epsilon}
\]

### Sweep

\[
P_{sweep}=\sigma(k_1 \cdot LiquidityDensity + k_2\cdot Volatility + k_3\cdot Momentum)
\]

### FOMO

\[
FOMO=\sigma(a_1 Momentum + a_2 DistanceFromRange + a_3 Breakout + a_4 LiquidityProximity)
\]

### Marge / levier

- Valeur position = \(Prix\times Taille\)
- Marge utilisée = \(Valeur/Levier\)
- Marge libre = \(Equity - Marge\)

Leviers supportés: `1:1`, `1:10`, `1:50`, `1:100`, `1:200`.

## Note méthodologique

Le moteur est **explicite et paramétrique**: aucune logique « IA boîte noire ». Les paramètres sont calibrés depuis les statistiques historiques, puis appliqués dans une simulation stochastique structurée.
