# Trading Bot — Documentation Complète

> Bot de trading algorithmique autonome : backtesting, trading papier et trading en direct avec gestion du risque avancée, IA intégrée et dashboard temps réel.

**AVERTISSEMENT** : Ce bot peut perdre de l'argent réel. Lisez ce guide en entier et complétez la checklist LIVE avant de risquer du capital.

---

## Table des matières

1. [Ce que fait le bot](#1-ce-que-fait-le-bot)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Modes de fonctionnement](#6-modes-de-fonctionnement)
7. [Stratégies de trading](#7-stratégies-de-trading)
8. [Gestion du risque](#8-gestion-du-risque)
9. [Détection de régime de marché](#9-détection-de-régime-de-marché)
10. [Intégrations IA et news](#10-intégrations-ia-et-news)
11. [Dashboards et API REST](#11-dashboards-et-api-rest)
12. [Alertes](#12-alertes)
13. [Déploiement en production](#13-déploiement-en-production)
14. [Tests](#14-tests)
15. [Dépannage](#15-dépannage)
16. [Avertissements légaux](#16-avertissements-légaux)

---

## 1. Ce que fait le bot

Le bot est un système de trading algorithmique complet. Il exécute un pipeline de décision toutes les 5 minutes :

```
Données marché → Indicateurs techniques → Régime de marché
       → Stratégies actives → Agrégation des signaux
       → Overlay news/IA → Règles métier → Gestionnaire de risque
       → Exécution (papier ou live)
```

### Capacités principales

| Fonctionnalité | Description |
|---|---|
| **Backtesting** | Test sur 5 ans d'historique avec validation walk-forward |
| **Trading papier** | Simulation en conditions réelles, sans risque |
| **Trading live** | Exécution réelle via Alpaca |
| **Multi-stratégie** | 7 stratégies actives simultanément selon le régime |
| **Multi-horizon** | Long terme (DCA mensuel) + Swing (1-10j) + Intraday (5min) |
| **Multi-actif** | ETFs US, actions tech, or, obligations, Forex (EUR/USD) |
| **Analyse IA** | Analyse quotidienne par Groq (LLaMA 3.3 70B) — gratuit |
| **Advisory IA** | Second avis optionnel via Claude Anthropic — payant |
| **News en temps réel** | Finnhub + flux RSS, impact sur la confiance des signaux |
| **Kill switch** | Arrêt automatique en cas de perte > seuil défini |
| **Dashboard backtest** | Interface web interactive avec sliders de paramètres |
| **Dashboard live** | Monitoring temps réel du portefeuille |
| **API REST** | 20+ endpoints FastAPI pour intégrations externes |
| **Alertes** | Notifications Telegram, Discord, Email |

---

## 2. Architecture du projet

```
trading_bot/
├── config/                     # Fichiers de configuration YAML
│   ├── settings.yaml           # Mode, données, broker, IA, alertes
│   ├── risk.yaml               # Limites de risque et kill switch
│   ├── strategies.yaml         # Paramètres des stratégies et régimes
│   └── profile.yaml            # Profil client (capital, tolérance, horizon)
│
├── src/
│   ├── main.py                 # Point d'entrée principal
│   ├── ai/                     # Couche IA
│   │   ├── advisory.py         # Advisory Anthropic Claude (optionnel)
│   │   ├── market_analyst.py   # Analyse quotidienne Groq LLaMA
│   │   └── strategic_planner.py
│   ├── strategies/             # 14 stratégies (7 actives)
│   │   ├── true_dca.py         # DCA long terme
│   │   ├── trend_following.py  # Suivi de tendance EMA + ADX
│   │   ├── breakout.py         # Cassure N-jours haut/bas
│   │   ├── rsi_dip_buyer.py    # RSI(2) mean reversion
│   │   ├── intraday_ema_cross.py       # Croisement EMA 5min
│   │   ├── intraday_bollinger_rsi.py   # Bollinger + RSI 5min
│   │   └── intraday_session_breakout.py # Cassure de session 5min
│   ├── engine/                 # Pipeline de décision
│   │   ├── decision_engine.py  # Orchestre régime → stratégies → risque
│   │   └── signal_aggregator.py # Consolide les signaux par actif
│   ├── features/               # Indicateurs techniques et régimes
│   │   ├── indicators.py       # EMA, RSI, ADX, Bollinger, ATR, MACD…
│   │   ├── feature_engine.py   # Calcule tous les indicateurs par actif
│   │   └── regime_detector.py  # Classification ML + règles du régime
│   ├── risk/
│   │   └── risk_manager.py     # Kill switch, sizing, limites d'exposition
│   ├── rules/
│   │   └── rules_engine.py     # Filtres statistiques et stratégiques
│   ├── execution/
│   │   ├── paper_broker.py     # Simulation locale (sans API)
│   │   └── alpaca_paper_broker.py # Intégration Alpaca paper
│   ├── backtesting/
│   │   ├── backtester.py       # Moteur + walk-forward (5 folds)
│   │   └── metrics.py          # Sharpe, drawdown, win rate, profit factor
│   ├── dashboard/
│   │   ├── backtest_dashboard.py # Dashboard backtest interactif
│   │   └── live_dashboard.py   # Monitoring live temps réel
│   ├── watchers/               # Threads de surveillance en fond
│   │   ├── market_watcher.py   # Orchestrateur central multi-fréquence
│   │   ├── portfolio_watcher.py # Positions et limites de risque
│   │   └── universe_builder.py # Gestion de l'univers d'actifs
│   ├── data/
│   │   ├── market_data.py      # Récupération et cache des données
│   │   └── yfinance_helpers.py
│   ├── news/
│   │   ├── news_manager.py     # Orchestrateur news
│   │   ├── collector.py        # Collecte Finnhub + flux RSS
│   │   └── classifier.py       # Score d'impact par actif
│   ├── alerts/
│   │   └── alert_manager.py    # Telegram, Discord, Email
│   ├── monitoring/
│   │   └── logger.py           # Logs structurés JSON + rapport d'audit
│   ├── portfolio/
│   │   └── allocation_engine.py # Allocation dynamique par stratégie
│   ├── profile/
│   │   └── client_profile.py   # Profil risque, objectifs, horizons
│   └── ml/
│       └── regime_model.py     # Modèle ML de prédiction de régime
│
├── api/                        # API REST FastAPI
│   ├── main.py                 # App FastAPI + lifespan
│   ├── models.py               # Schémas Pydantic
│   ├── dependencies.py         # État partagé
│   └── routes/
│       ├── portfolio.py        # GET /portfolio/*
│       ├── decisions.py        # GET /decisions/*
│       ├── news.py             # GET /news/*
│       └── profile.py          # GET/PUT /profile/*
│
├── tests/
│   └── test_core.py            # 39 tests unitaires
│
├── docs/
│   └── LIVE_CHECKLIST.md       # Checklist pré-live (9 sections, 80+ points)
│
├── data/
│   ├── cache/                  # Cache yfinance (TTL 5 min)
│   ├── logs/                   # Logs JSON structurés
│   └── reports/                # Rapports quotidiens/hebdomadaires
│
├── requirements.txt
├── Procfile                    # Déploiement Render.com
└── .env.example                # Modèle de variables d'environnement
```

---

## 3. Prérequis

### Système

- **Windows 10/11**, macOS ou Linux
- **Python 3.10 ou supérieur** — [Télécharger](https://www.python.org/downloads/)
- **Git** — [Télécharger](https://git-scm.com/)

### Comptes externes

| Service | Obligatoire | Usage |
|---|---|---|
| **Alpaca** | Oui (pour live) | Broker pour exécuter les ordres réels |
| **Finnhub** | Recommandé | News financières temps réel |
| **Groq** | Recommandé | Analyse IA quotidienne (gratuit) |
| **Anthropic** | Optionnel | Advisory IA avancé (payant) |
| **Telegram / Discord** | Optionnel | Alertes en temps réel |

---

## 4. Installation

### Étape 1 — Cloner le projet

```bash
git clone https://github.com/votre-username/trading_bot.git
cd trading_bot
```

### Étape 2 — Créer un environnement virtuel

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Vous devez voir `(.venv)` au début de votre invite de commande.

### Étape 3 — Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 4 — Vérifier l'installation

```bash
python -m pytest tests/test_core.py -q
```

Résultat attendu : `39 passed`. Si tous les tests passent, l'installation est correcte.

### Étape 5 — Configurer les secrets

Copiez le fichier d'exemple et remplissez vos clés :

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Ensuite éditez `.env` :

```bash
# ── BROKER (Alpaca) ─────────────────────────────────────────────
BROKER_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
BROKER_API_SECRET=yyyyyyyyyyyyyyyyyyyyyy
# Pour paper trading :
BROKER_BASE_URL=https://paper-api.alpaca.markets/v2
# Pour trading réel (UNIQUEMENT après validation complète) :
# BROKER_BASE_URL=https://api.alpaca.markets/v2

# ── IA (optionnel) ───────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx   # Claude advisory
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # Analyse quotidienne

# ── NEWS ─────────────────────────────────────────────────────────
FINNHUB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

# ── ALERTES (optionnel) ──────────────────────────────────────────
TELEGRAM_BOT_TOKEN=123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef
TELEGRAM_CHAT_ID=987654321
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/xxx/yyy

# ── EMAIL (optionnel) ────────────────────────────────────────────
EMAIL_USER=votre@gmail.com
EMAIL_PASSWORD=votre_mot_de_passe_app
EMAIL_RECIPIENT=alerts@votre-domaine.com

# ── SYSTÈME (optionnel) ──────────────────────────────────────────
LOG_LEVEL=INFO
CACHE_TTL_SECONDS=300
LOG_DIR=data/logs
```

**Ne committez jamais ce fichier `.env`** — il est déjà dans `.gitignore`.

---

## 5. Configuration

Tous les paramètres sont dans `config/` au format YAML. Voici les options importantes :

### `config/settings.yaml` — Configuration globale

```yaml
system:
  mode: "paper"          # "paper" ou "live" (commencez toujours par paper)
  live_enabled: false    # Verrou de sécurité — mettre true UNIQUEMENT pour le live
  timezone: "Europe/Paris"

data:
  provider: "yfinance"
  intraday_interval: "5m"    # Intervalle pour les stratégies intraday
  cache_ttl_seconds: 300     # Cache de 5 minutes pour les données

universe:
  long_term: ["SPY", "QQQ", "IWM", "GLD", "TLT", "VT"]   # ETFs DCA
  swing: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XLE"]
  intraday: ["EURUSD=X"]    # Forex 5min

broker:
  paper:
    initial_capital: 10000.0    # Capital de départ (USD)
    commission_per_trade: 0.0
    slippage_pct: 0.0002        # 2 pips de slippage simulé

ai:
  enabled: false               # Advisory Anthropic (payant, optionnel)
  advisory_only: true          # JAMAIS de trade automatique via l'IA

market_analyst:
  provider: "groq"             # Analyse quotidienne gratuite
  model: "llama-3.3-70b-versatile"
  enabled: true

news:
  enabled: true
  max_age_hours: 24
  refresh_interval_minutes: 30

market_watcher:
  mvp_mode: true               # Cycle unique de 5 min (MVP)
```

### `config/risk.yaml` — Limites de risque

```yaml
risk:
  max_risk_per_trade_pct: 0.005        # 0.5% du capital par trade max
  max_daily_loss_pct: 0.01             # Arrêt à -1% sur la journée
  max_weekly_loss_pct: 0.03            # Arrêt à -3% sur la semaine
  max_monthly_drawdown_pct: 0.06       # Arrêt à -6% sur le mois
  max_total_drawdown_pct: 0.15         # Mode défensif à -15%
  max_open_positions: 15
  max_exposure_per_asset_pct: 0.10     # Max 10% sur un seul actif
  max_exposure_per_sector_pct: 0.25    # Max 25% sur un secteur
  max_correlation_exposure: 0.70

kill_switch:
  daily_loss_pct: 0.01                 # Déclenchement automatique à -1%
  halt_duration_minutes: 0             # 0 = redémarrage manuel obligatoire

position_sizing:
  method: "fixed_fractional"           # "fixed_fractional", "kelly", "equal_weight"
  kelly_fraction: 0.25                 # Quarter-Kelly (conservateur)
```

### `config/strategies.yaml` — Paramètres des stratégies

Ce fichier contrôle les paramètres de chaque stratégie et définit quelle stratégie est active selon le régime de marché.

```yaml
# Exemple : paramètres Trend Following
trend_following:
  enabled: true
  min_adx: 25                  # ADX minimum pour confirmer la tendance
  atr_multiplier_sl: 2.0       # Stop loss = 2x ATR
  atr_multiplier_tp: 3.0       # Take profit = 3x ATR

# Mapping régime → stratégies actives
regime_strategy_map:
  bull_trend:
    - true_dca
    - trend_following
    - breakout
    - rsi_dip_buyer
    - intraday_ema_cross
    - intraday_session_breakout
  bear_trend:
    - true_dca
    - intraday_ema_cross
  panic:
    - true_dca          # Seulement DCA, pas d'intraday en panique
```

### `config/profile.yaml` — Profil investisseur

```yaml
client:
  capital: 10000.0
  risk_tolerance: "moderate"         # conservative | moderate | aggressive | speculative
  objective: "growth"                # income | growth | wealth_preservation | balanced
  horizon_years: 5
  max_drawdown_tolerance: 0.20       # Tolérance personnelle au drawdown

  preferences:
    intraday: true        # Active le trading 5min sur EUR/USD
    forex: true
```

---

## 6. Modes de fonctionnement

### 6.1 — Backtesting

Testez une stratégie sur les données historiques (5 ans par défaut) :

```bash
# Backtest d'une stratégie sur SPY
python -m src.main --backtest SPY --strategy trend_following

# Backtest sur plusieurs actifs
python -m src.main --backtest QQQ --strategy rsi_dip_buyer

# Backtest DCA long terme
python -m src.main --backtest SPY --strategy true_dca
```

**Métriques produites :**

| Métrique | Description | Objectif |
|---|---|---|
| Rendement total | Performance sur la période | Le plus haut possible |
| Sharpe ratio | Rendement ajusté au risque | > 0.8 |
| Max drawdown | Perte maximale depuis un pic | < 15% |
| Win rate | % de trades gagnants | > 40% |
| Profit factor | Gains bruts / Pertes brutes | > 1.3 |
| Walk-forward OOS | Validation hors-échantillon (5 folds) | > 70% |

### 6.2 — Dashboard Backtest Interactif

Interface web pour visualiser et ajuster les backtests en temps réel :

```bash
python -m src.dashboard.backtest_dashboard
```

Ouvrez ensuite `http://localhost:8050/`

**Fonctionnalités du dashboard :**

- Graphique de la courbe de capital (equity curve)
- Drawdown par période
- Rendements annuels
- Tableau de tous les trades (entrée, sortie, taille, P&L)
- Sélecteur de stratégie (true_dca, trend_following, breakout, rsi_dip_buyer)
- Sliders pour ajuster chaque paramètre en direct :
  - DCA : `monthly_size_pct`, `dip_size_pct`, `dip_threshold_pct`
  - Trend Following : `min_adx`, `atr_multiplier_sl`, `atr_multiplier_tp`
  - Breakout : `lookback_period`, `atr_multiplier_sl`, `atr_multiplier_tp`
  - RSI Dip Buyer : `rsi2_entry_threshold`, `position_size_pct`
- Bouton "Re-run" pour recalculer instantanément
- Export CSV des résultats

### 6.3 — Trading Papier (obligatoire 30+ jours)

Simule le trading live avec des données de marché réelles, sans capital réel :

```bash
python -m src.main --mode paper
# ou simplement :
python -m src.main
```

Le bot tourne en boucle. À chaque cycle de 5 minutes :

1. Récupère les données de marché via yfinance
2. Calcule les indicateurs techniques
3. Détecte le régime de marché actuel
4. Active les stratégies appropriées
5. Génère des signaux d'achat/vente avec score de confiance
6. Applique l'overlay news (si Finnhub activé)
7. Consulte l'IA Groq pour l'analyse du jour
8. Filtre via le moteur de règles
9. Valide via le gestionnaire de risque
10. **Simule** l'exécution (sans vrai ordre)
11. Enregistre tout dans les logs

**Logs et rapports :**
- `data/logs/` — Logs JSON structurés de chaque décision
- `data/reports/` — Rapport quotidien automatique
- Alertes Telegram/Discord si configurées

**Checklist avant de passer en live** (voir [docs/LIVE_CHECKLIST.md](docs/LIVE_CHECKLIST.md)) :
- 30+ jours sans crash ni erreur critique
- Sharpe ratio > 0.8 en papier
- Max drawdown < 15%
- Win rate > 40%, Profit factor > 1.3
- Toutes les limites de risque testées individuellement
- Alertes fonctionnelles (Telegram/Discord)
- Infrastructure sécurisée (pas de clés en dur, logs rotatifs)

### 6.4 — Trading en Direct

**Uniquement après avoir complété la checklist LIVE en entier.**

**Étape 1 — Modifier la configuration :**

```yaml
# config/settings.yaml
system:
  mode: "live"           # Changer de "paper"
  live_enabled: true     # Lever le verrou de sécurité
```

**Étape 2 — Pointer vers l'API live Alpaca dans `.env` :**

```bash
BROKER_BASE_URL=https://api.alpaca.markets/v2   # URL LIVE (pas paper!)
```

**Étape 3 — Démarrer :**

```bash
python -m src.main --mode live
```

**Recommandations pour les 30 premiers jours en live :**
- Commencez avec 10% du capital prévu
- Utilisez des limites de risque moitié moins élevées (`max_risk: 0.0025`, `max_daily_loss: 0.005`)
- Augmentez progressivement si les performances sont conformes aux backtests

### 6.5 — API REST

Expose l'état du portefeuille, les décisions et les news à des applications externes :

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation interactive : `http://localhost:8000/docs`

**Endpoints disponibles :**

```
GET  /health                        État du système
GET  /portfolio/state               État complet du portefeuille
GET  /portfolio/positions           Positions ouvertes
GET  /portfolio/performance         Métriques de performance
GET  /decisions/recent              Décisions récentes
GET  /decisions/history             Historique complet
GET  /news/latest                   Dernières news
GET  /news/impact/{symbol}          Impact news sur un actif
GET  /profile                       Profil client
PUT  /profile                       Modifier le profil
```

---

## 7. Stratégies de trading

### 7.1 — True DCA (Dollar Cost Averaging)

| Paramètre | Valeur | Description |
|---|---|---|
| Horizon | Long terme | Mensuel |
| Actifs | SPY, QQQ, IWM, GLD, TLT, VT | ETFs diversifiés |
| Allocation mensuelle | 5% du capital | Achat régulier |
| Dip buying | Oui | +7.5% si baisse > 5% |
| Exposition max | 85% du portefeuille | Garde toujours 15% en cash |

**Comment ça marche :** Le bot achète 5% du capital en ETFs chaque mois, et augmente l'allocation à 7.5% si les prix ont baissé de plus de 5%. C'est la stratégie la plus sûre, active dans tous les régimes.

---

### 7.2 — Trend Following

| Paramètre | Valeur | Description |
|---|---|---|
| Horizon | Swing | 1 à 10 jours |
| Actifs | AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XLE | Actions |
| Entrée | EMA20 > EMA50 > EMA200 + ADX > 25 | Tendance haussière confirmée |
| Stop loss | 2× ATR | Adaptatif à la volatilité |
| Take profit | 3× ATR | Ratio risque/récompense 1:1.5 |

**Comment ça marche :** Entre en position quand les moyennes mobiles sont alignées à la hausse ET que l'ADX confirme une tendance forte. Sort automatiquement via stop ATR.

---

### 7.3 — Breakout

| Paramètre | Valeur | Description |
|---|---|---|
| Horizon | Swing | 1 à 10 jours |
| Lookback | 20 jours | Calcul du range |
| Filtre | EMA200 | Seuls les breakouts dans la direction de la tendance longue |
| Stop loss | 1.5× ATR | En dessous du niveau cassé |
| Take profit | 2.5× ATR | |

**Comment ça marche :** Identifie les cassures au-dessus du plus haut des 20 derniers jours (ou en dessous du plus bas pour short). Filtre par alignement avec l'EMA200 pour éviter les faux breakouts.

---

### 7.4 — RSI Dip Buyer (Mean Reversion)

| Paramètre | Valeur | Description |
|---|---|---|
| Horizon | Swing | 2 à 5 jours |
| Signal d'entrée | RSI(2) < 15 | Survente extrême |
| Signal de sortie | RSI(2) > 65 | Retour à la moyenne |
| Stop loss | 2× ATR | Protection contre tendance baissière |

**Comment ça marche :** Le RSI sur 2 périodes est très sensible aux mouvements courts. Un RSI(2) < 15 indique une survente extrême à court terme. Le bot entre en position et attend le rebond vers la moyenne.

---

### 7.5 — Intraday EMA Cross (5min — EUR/USD)

| Paramètre | Valeur | Description |
|---|---|---|
| Horizon | Intraday | Minutes à quelques heures |
| Actif principal | EUR/USD | Forex |
| Signal | EMA(9) croise EMA(21) + RSI(14) 40-60 | |
| Filtre | RSI hors des extrêmes | Évite les zones de surachat/survente |

**Comment ça marche :** Croisement classique de moyennes mobiles rapide/lente sur le graphique 5 minutes, avec filtre RSI pour éviter les faux signaux en zones extrêmes.

---

### 7.6 — Intraday Bollinger RSI (5min — range trading)

| Paramètre | Valeur | Description |
|---|---|---|
| Bollinger Bands | (20, 2.0) | Calcul sur 20 bougies |
| Entrée achat | Prix < bande basse + RSI(14) < 30 | Double confirmation |
| Entrée vente | Prix > bande haute + RSI(14) > 70 | |
| Cible | Bande médiane | Retour à la moyenne |

**Comment ça marche :** Stratégie de range trading. Quand le prix touche la bande basse des Bollinger ET que le RSI confirme la survente, le bot entre long en anticipant un retour vers le milieu des bandes.

---

### 7.7 — Intraday Session Breakout (5min — cassures de session)

| Paramètre | Valeur | Description |
|---|---|---|
| Sessions | Londres + New York | Horaires fixes |
| Range | Calculé sur les 30 premières minutes | Haut/bas d'ouverture |
| Signal | Cassure du range de session + volume | |

**Comment ça marche :** Les premières 30 minutes de chaque session établissent un range. La cassure de ce range (haut ou bas) avec du volume déclenche un trade dans la direction de la cassure.

---

### Activation des stratégies par régime

| Régime | Stratégies actives |
|---|---|
| `bull_trend` | true_dca, trend_following, breakout, rsi_dip_buyer, intraday_ema_cross, session_breakout |
| `bear_trend` | true_dca, intraday_ema_cross |
| `range` | true_dca, rsi_dip_buyer, bollinger_rsi, intraday_ema_cross |
| `high_volatility` | true_dca, rsi_dip_buyer |
| `panic` | true_dca uniquement |
| `compression` | true_dca, bollinger_rsi, session_breakout |
| `euphoric` | true_dca, breakout (réduction de taille) |

---

## 8. Gestion du risque

Le gestionnaire de risque est le **dernier verrou** avant l'exécution. Il a un droit de veto absolu sur chaque ordre.

### Limites et actions

| Limite dépassée | Action |
|---|---|
| Risque par trade > 0.5% | Réduction automatique de la taille |
| Exposition actif > 10% | Blocage du trade |
| Perte quotidienne > 1% | Arrêt de tous les trades de la journée |
| Perte hebdomadaire > 3% | Arrêt jusqu'à la semaine suivante |
| Drawdown > 5% | Mode défensif (exposition réduite de 50%) |
| Drawdown total > 15% | Kill switch déclenché |

### Kill Switch

Quand le kill switch se déclenche :
1. Tous les nouveaux ordres sont bloqués
2. Une alerte critique est envoyée (Telegram/Discord/Email)
3. Le bot attend un **redémarrage manuel** (paramètre `halt_duration_minutes: 0`)
4. Investiguer les logs avant de redémarrer

### Sizing des positions

Trois méthodes disponibles dans `config/risk.yaml` :

- **`fixed_fractional`** : Taille fixe = % du capital (recommandé pour débuter)
- **`kelly`** : Kelly fraction = 25% (quarter-Kelly, conservateur)
- **`equal_weight`** : Poids égal entre toutes les positions

---

## 9. Détection de régime de marché

Le bot identifie 9 régimes de marché et adapte ses stratégies en conséquence.

| Régime | Description | Indicateurs clés |
|---|---|---|
| `bull_trend` | Tendance haussière forte | EMA alignées, ADX > 25, VIX bas |
| `bear_trend` | Tendance baissière | EMA baissières, momentum négatif |
| `range` | Marché latéral | ADX < 20, prix entre supports/résistances |
| `high_volatility` | Forte volatilité | VIX élevé, ATR en hausse |
| `low_volatility` | Volatilité comprimée | VIX bas, Bollinger étroit |
| `panic` | Vente panique | Chute rapide, RSI extrême bas |
| `euphoric` | Euphorie acheteuse | RSI extrême haut, momentum fort |
| `compression` | Compression (avant explosion) | Bollinger très étroit |
| `breakout_expansion` | Expansion après compression | Bollinger s'écarte |

La détection utilise une combinaison de règles fixes + modèle ML (`src/ml/regime_model.py`).

---

## 10. Intégrations IA et news

### Groq — Analyse de marché quotidienne (gratuit)

- **Modèle** : LLaMA 3.3 70B via l'API Groq
- **Fréquence** : 1× par jour
- **Inputs** : Régime actuel, indicateurs, news
- **Output** : Résumé des conditions, risques clés, opportunités identifiées
- **Activation** : `market_analyst.enabled: true` dans `settings.yaml`
- **Coût** : Gratuit (quota Groq)

### Anthropic Claude — Advisory IA (optionnel, payant)

- **Modèles** :
  - Claude Opus (temp=0.2) pour l'évaluation stratégique de début de journée
  - Claude Sonnet (temp=0.6) pour l'analyse en temps réel
- **Rôle** : Second avis consultatif uniquement — **jamais d'exécution automatique**
- **Output** : `opportunity_score`, `risk_score`, `agreement`, `warnings`, `recommended_action`
- **Activation** : `ai.enabled: true` dans `settings.yaml`
- **Important** : `ai.advisory_only: true` est toujours forcé

### Finnhub — News en temps réel

- **Données** : News d'entreprises, alertes résultats, scores de sentiment
- **Fréquence** : Collecte toutes les 30 minutes
- **Impact** : Modifie le score de confiance des signaux par actif
- **Exemple** : News négative sur NVDA → réduction de la confiance du signal NVDA

### Flux RSS

- Complète Finnhub avec des sources RSS financières configurables
- Classifie automatiquement l'impact par actif et secteur

---

## 11. Dashboards et API REST

### Dashboard Backtest — `http://localhost:8050`

```bash
python -m src.dashboard.backtest_dashboard
```

Interface Dash/Plotly interactive :
- Courbe de capital + drawdown + rendements annuels
- Tableau des trades avec filtre et tri
- Sliders temps réel pour tous les paramètres
- Comparaison de stratégies
- Export CSV

### Dashboard Live — `http://localhost:8050`

```bash
python -m src.dashboard.live_dashboard
```

Monitoring temps réel :
- Portefeuille total + cash + P&L ouvert
- Liste des positions actives
- Régime de marché courant + VIX
- Décisions récentes + statut d'exécution
- Feed news en direct
- Métriques de risque (drawdown quotidien/hebdomadaire/mensuel)

### API REST — `http://localhost:8000`

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation Swagger : `http://localhost:8000/docs`

---

## 12. Alertes

Configurez les alertes dans `config/settings.yaml` et `.env`.

### Telegram

```yaml
# config/settings.yaml
alerts:
  telegram:
    enabled: true
    bot_token_env: "TELEGRAM_BOT_TOKEN"
    chat_id_env: "TELEGRAM_CHAT_ID"
```

**Types d'alertes envoyées :**
- Signal d'achat/vente exécuté
- Kill switch déclenché (critique)
- Limite de perte dépassée
- Rapport quotidien

### Discord

```yaml
alerts:
  discord:
    enabled: true
    webhook_url_env: "DISCORD_WEBHOOK_URL"
```

### Email

```yaml
alerts:
  email:
    enabled: true
    user_env: "EMAIL_USER"
    password_env: "EMAIL_PASSWORD"
    recipient_env: "EMAIL_RECIPIENT"
```

---

## 13. Déploiement en production

### Sur Render.com (dashboard live)

Le `Procfile` configure le déploiement automatique :

```
web: python -m src.dashboard.live_dashboard --host 0.0.0.0 --port $PORT
```

**Étapes :**

1. Poussez le code sur GitHub
2. Créez un nouveau service sur [render.com](https://render.com)
3. Connectez votre dépôt GitHub
4. Ajoutez toutes les variables d'environnement dans le panneau Render
5. Render déploie automatiquement à chaque push sur `main`

### Sur un VPS / serveur dédié (bot de trading)

Pour faire tourner le bot en production sur un serveur Linux :

**Option 1 — tmux (simple)**

```bash
tmux new -s trading_bot
source .venv/bin/activate
python -m src.main --mode live
# Ctrl+B puis D pour détacher
```

**Option 2 — systemd (recommandé en production)**

Créez `/etc/systemd/system/trading-bot.service` :

```ini
[Unit]
Description=Trading Bot
After=network.target

[Service]
User=votre_user
WorkingDirectory=/chemin/vers/trading_bot
ExecStart=/chemin/vers/.venv/bin/python -m src.main --mode live
Restart=on-failure
RestartSec=30
EnvironmentFile=/chemin/vers/trading_bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

### Variables d'environnement Render

Ajoutez ces variables dans le panneau "Environment" de Render :

```
BROKER_API_KEY
BROKER_API_SECRET
BROKER_BASE_URL
ANTHROPIC_API_KEY
GROQ_API_KEY
FINNHUB_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

---

## 14. Tests

### Lancer tous les tests

```bash
python -m pytest tests/test_core.py -v
```

### Tests inclus (39 au total)

| Module testé | Tests |
|---|---|
| Indicateurs techniques | EMA, RSI, ATR, ADX, Bollinger, MACD |
| Stratégies | Génération de signaux pour chaque stratégie |
| Moteur de règles | Validation des filtres statistiques et stratégiques |
| Gestionnaire de risque | Kill switch, sizing, limites d'exposition |

### Lancer avec couverture de code

```bash
python -m pytest tests/test_core.py --cov=src --cov-report=html
# Rapport dans htmlcov/index.html
```

---

## 15. Dépannage

### `ModuleNotFoundError: No module named 'src'`

```bash
# Vérifiez que vous êtes dans le bon dossier
cd c:\Users\emili\Documents\trading_bot

# Activez l'environnement virtuel
.venv\Scripts\Activate.ps1

# Réinstallez les dépendances
pip install -r requirements.txt
```

### `Port 8050 already in use`

```bash
# Windows — trouvez et tuez le processus
netstat -ano | findstr :8050
taskkill /PID [PID] /F

# Ou utilisez un autre port
python -m src.dashboard.backtest_dashboard --port 8051
```

### `No data available` ou `yfinance rate limit`

- Attendez 1-2 minutes (yfinance a un rate limit)
- Réduisez le nombre de symboles testés
- Augmentez `cache_ttl_seconds` dans `settings.yaml`

### `BROKER_API_KEY not found`

1. Vérifiez que le fichier `.env` existe à la racine du projet
2. Vérifiez que les clés sont correctement remplies
3. Redémarrez le bot

### `Kill switch triggered` — le bot s'est arrêté

1. Consultez les logs dans `data/logs/`
2. Identifiez la cause (perte limite, erreur critique)
3. Corrigez le problème
4. Relancez manuellement : `python -m src.main`

### Tests qui échouent

```bash
# Réinstallez les dépendances
pip install --force-reinstall -r requirements.txt
python -m pytest tests/test_core.py -q --tb=short
```

---

## 16. Avertissements légaux

- **Aucune garantie** : Les performances passées ne garantissent pas les résultats futurs
- **Risque de perte totale** : Ce bot peut perdre 100% du capital investi
- **Pas de conseil financier** : Ce projet est éducatif. Consultez un professionnel avant de trader avec du capital réel
- **Conformité locale** : Vérifiez que le trading algorithmique est légal dans votre pays
- **Fiscalité** : Le trading fréquent a des implications fiscales importantes selon votre juridiction

---

*La gestion du risque est la priorité absolue. Ne désactivez jamais le Risk Manager.*
