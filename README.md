# 🤖 Trading Bot — Guide Complet d'Installation et Utilisation

Un bot de trading algorithme autonome avec **backtesting**, **trading papier**, et **trading en direct**. Conçu pour minimiser le risque avec des limites strictes et une gestion de portefeuille intelligente.

**⚠️ AVERTISSEMENT IMPORTANT** : Ce bot peut perdre de l'argent réel. Lisez entièrement ce guide et la checklist LIVE avant de risquer du capital.

---

## 📋 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Prérequis](#-prérequis)
3. [Installation](#-installation)
4. [Configuration](#-configuration)
5. [Utilisation](#-utilisation)
6. [Structure du projet](#-structure-du-projet)
7. [Stratégies disponibles](#-stratégies-disponibles)
8. [Dépannage](#-dépannage)

---

## 🎯 Vue d'ensemble

Le Trading Bot offre trois modes de fonctionnement :

### 1. **Backtesting** 📊
- Testez vos stratégies sur 5+ années de données historiques
- Simulez commissions et slippage réalistes
- Générez des rapports détaillés (Sharpe ratio, drawdown, win rate)
- Utilisez le dashboard interactif pour visualiser les résultats

### 2. **Trading Papier** 📝
- Simulez le trading en direct avec votre compte réel
- Testez les stratégies dans des conditions réelles SANS risque
- Validez que tous les systèmes fonctionnent avant le live
- Obligatoire : 30+ jours avant de passer au trading réel

### 3. **Trading en Direct** 💰
- Trading automatique avec capital réel
- Intégration broker (Alpaca, IBKR)
- Gestion automatique du risque : arrêt du trading si pertes > seuil
- Alertes Telegram/Discord en temps réel

---

## ✅ Prérequis

### Système d'exploitation
- **Windows** (recommandé), **macOS**, ou **Linux**

### Logiciels requis
- **Python 3.10+** ([Télécharger](https://www.python.org/downloads/))
- **Git** ([Télécharger](https://git-scm.com/))
- (Optionnel) **Docker** pour la production

### Comptes externes
- **Compte Alpaca** ou **IBKR** (pour le trading en direct)
- **API key Anthropic** (pour l'IA) — optionnel mais recommandé

---

## 🚀 Installation

### Étape 1 : Cloner le projet

```bash
# Ouvrez un terminal/PowerShell et exécutez :
git clone https://github.com/[votre-username]/trading_bot.git
cd trading_bot
```

### Étape 2 : Créer un environnement Python virtuel

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux (Bash)
python3 -m venv .venv
source .venv/bin/activate
```

**Vous devez voir `(.venv)` au début de votre ligne de commande.**

### Étape 3 : Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 4 : Vérifier l'installation

```bash
# Testez que tout fonctionne :
python -m pytest tests/test_core.py -q
```

✅ Si vous voyez `39 passed`, c'est bon !

### Étape 5 : Configurer les secrets (variables d'environnement)

Créez un fichier `.env` à la racine du projet (à côté de `requirements.txt`) :

```bash
# Variables Broker (pour le trading en direct)
BROKER_API_KEY=your_alpaca_api_key_here
BROKER_API_SECRET=your_alpaca_secret_here
BROKER_BASE_URL=https://paper-api.alpaca.markets  # ou live URL

# Variables IA (optionnel, mais recommandé)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Variables Alertes (optionnel)
TELEGRAM_BOT_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**⚠️ Ne committez JAMAIS ce fichier.** Il est automatiquement ignoré par Git.

---

## ⚙️ Configuration

Tous les fichiers de configuration sont dans le dossier `config/` en format **YAML**.

### Fichier 1 : `config/settings.yaml`

Contrôle le **mode de fonctionnement** et les **données**.

```yaml
system:
  mode: "paper"              # ← Changez à "live" APRÈS validation complète
  live_enabled: false        # ← Doit être true pour le trading réel
  timezone: "Europe/Paris"   # ← Changez selon votre timezone

broker:
  paper:
    initial_capital: 500.0   # ← Capital de départ en USD
    commission_per_trade: 0.0
    slippage_pct: 0.001      # 0.1% de slippage simulé
```

**Actions recommandées** :
- Pour tester : Gardez `mode: paper`
- Pour backtesting : Gardez `mode: paper`
- Pour le trading réel : Changez à `mode: live` SEULEMENT après la checklist complète

### Fichier 2 : `config/risk.yaml`

Définis les **limites de risque** (très important !).

```yaml
risk:
  max_risk_per_trade_pct: 0.005       # Max 0.5% par trade
  max_daily_loss_pct: 0.01            # Le bot s'arrête à -1%
  max_weekly_loss_pct: 0.03           # Le bot s'arrête à -3%
  max_monthly_drawdown_pct: 0.06      # Le bot s'arrête à -6%
  
  # Limites de position
  max_open_positions: 15              # Max 15 positions ouvertes
  max_exposure_per_asset_pct: 0.10    # Max 10% dans un seul titre

kill_switch:
  daily_loss_pct: 0.01                # Arrête tout le bot
  halt_duration_minutes: 0            # 0 = redémarrage manuel obligatoire
```

**Actions recommandées** :
- Pour débuter en live : Réduisez les limites de moitié (max_risk: 0.0025, max_daily_loss: 0.005)
- Après 30 jours : Augmentez progressivement si tout va bien

### Fichier 3 : `config/strategies.yaml`

Active/désactive les **stratégies** et définit leurs paramètres.

```yaml
strategies:
  tactical_dca:
    enabled: true               # ← Active cette stratégie
    assets: ["SPY", "QQQ", "GLD", "TLT"]
    frequency: "monthly"
    buy_day_of_month: 1         # Achat le 1er du mois
    dip_buy_enabled: true       # Acheter extra en cas de baisse
    dip_threshold_pct: -0.05    # Achat supplémentaire si -5%

  true_dca:
    enabled: true
    assets: ["SPY"]
    monthly_size_pct: 0.05      # 5% du capital par mois
    dip_size_pct: 0.075         # 7.5% en cas de baisse

  trend_following:
    enabled: true
    timeframe: "1d"             # Données journalières
```

**Actions recommandées** :
- Testez une seule stratégie d'abord : Désactivez-les sauf `tactical_dca`
- Activez progressivement après validation

---

## 📊 Utilisation

### 1. Backtesting (Tester une stratégie sur l'historique)

Backtestez une stratégie sur **5 années de données passées** :

```bash
python -m src.main \
  --symbols SPY QQQ \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --strategies tactical_dca trend_following
```

**Résultat** : Un rapport détaillé avec :
- Rendement total
- Sharpe ratio
- Max drawdown
- Win rate
- Nombre de trades

### 2. Dashboard Interactif 📈

Lancez le dashboard pour **visualiser** les backtests :

```bash
python -m src.dashboard.backtest_dashboard
```

Puis ouvrez : `http://localhost:8050/`

**Fonctionnalités** :
- Graphique de performance
- Tableau des positions
- Paramètres ajustables en temps réel
- Export des résultats

### 3. Trading Papier (30+ jours obligatoires)

Simulez le trading en direct avec votre compte réel :

```bash
python -m src.main --mode paper
```

**Comportement** :
- Récupère les données de marché en direct
- Lance les signaux de trading (sans vraiment exécuter)
- Simule les gains/pertes
- Génère des alertes et logs
- Réinitialise chaque jour

**Checklist avant de passer au live** (voir `docs/LIVE_CHECKLIST.md`) :
- ✅ 30+ jours de trading papier sans crash
- ✅ Aucune alerte critique
- ✅ Profit factor > 1.3
- ✅ Win rate > 40%
- ✅ Max drawdown < 15%

### 4. Trading en Direct 💰

**⚠️ SEULEMENT APRÈS la validation complète !**

#### Étape 1 : Configurer le broker

```yaml
# config/settings.yaml
broker:
  live:
    name: "alpaca"
    api_key_env: "BROKER_API_KEY"
    api_secret_env: "BROKER_API_SECRET"

system:
  mode: "live"           # ← Changez de "paper"
  live_enabled: true     # ← Mettez à true
```

#### Étape 2 : Configurer les secrets

```bash
# Ouvrez .env et remplissez :
BROKER_API_KEY=PKxxxxxx
BROKER_API_SECRET=yyyyyyyy
BROKER_BASE_URL=https://api.alpaca.markets  # URL LIVE (pas paper!)
```

#### Étape 3 : Démarrer le bot

```bash
python -m src.main --mode live
```

**Recommandations pour débuter** :
- Testez avec **10% seulement** du capital prévu pour les 30 premiers jours
- Gardez des **limites de risque conservatrices** (max_risk: 0.25%, max_daily_loss: 0.5%)
- **Augmentez progressivement** après 30 jours de succès

---

## 📁 Structure du projet

```
trading_bot/
├── config/                    # Fichiers de configuration
│   ├── settings.yaml         # Paramètres globaux
│   ├── risk.yaml            # Limites de risque
│   └── strategies.yaml       # Configuration des stratégies
│
├── src/
│   ├── main.py              # Point d'entrée principal
│   ├── engine/              # Moteur de décision
│   ├── execution/           # Exécution des trades
│   ├── backtesting/         # Moteur de backtest
│   ├── dashboard/           # Dashboard Dash/Plotly
│   ├── strategies/          # Toutes les stratégies
│   ├── risk/                # Gestionnaire de risque
│   ├── monitoring/          # Logs et alertes
│   └── data/                # Récupération données
│
├── data/
│   ├── cache/               # Cache des données yfinance
│   └── logs/                # Logs du bot (JSON)
│
├── tests/                   # Tests unitaires
│   └── test_core.py        # Tests principaux
│
├── docs/
│   └── LIVE_CHECKLIST.md    # Checklist avant trading réel ⚠️
│
├── requirements.txt         # Dépendances Python
├── README.md               # Ce fichier
└── .env.example            # Exemple de variables d'environnement
```

---

## 🎮 Stratégies disponibles

### 1. **Tactical DCA** (Dollar Cost Averaging tactique)
- **Type** : Long terme (horizon : mois)
- **Ideal pour** : Accumuler du capital progressivement
- **Comment ça marche** :
  - Achète 5% du portefeuille tous les mois
  - Achète 2x plus si le marché chute de 5%
  - Réduit les positions en marché baissier
- **Paramètres clés** : `buy_day_of_month`, `dip_threshold_pct`

### 2. **True DCA** (DCA pur)
- **Type** : Long terme (investissement)
- **Idéal pour** : Construcción graduelle de richesse
- **Comment ça marche** :
  - Alloue exactement 5% du capital par mois
  - Limite l'exposition totale à 85% du portefeuille
  - Maintient toujours 10% en cash

### 3. **Trend Following**
- **Type** : Swing (horizon : 1-10 jours)
- **Idéal pour** : Capturer les tendances à court terme
- **Comment ça marche** :
  - Achète quand l'EMA 20 croise l'EMA 50
  - Vend quand le prix revient sous l'EMA
  - Fonctionne sur SPY, QQQ, actions tech

### 4. **Mean Reversion**
- **Type** : Swing/intraday
- **Idéal pour** : Exploiter les rebonds
- **Comment ça marche** :
  - Identifie les extrema (RSI < 30 ou > 70)
  - Rentre en position au rebond
  - Cible le prix moyen

### 5. **Momentum**
- **Type** : Swing
- **Idéal pour** : Suivre les winners
- **Comment ça marche** :
  - Achète les titres en haut momentum
  - Laisse courir les gagnants
  - Stop loss si le momentum s'effondre

### 6. **Volatility Compression**
- **Type** : Swing
- **Idéal pour** : Anticiper les mouvements volatiles
- **Comment ça marche** :
  - Détecte quand la volatilité chute
  - Rentre en position juste avant une explosion
  - Beneficia des gros mouvements

### 7. **Breakout**
- **Type** : Swing
- **Idéal pour** : Capturer les cassures
- **Comment ça marche** :
  - Définit support/résistance sur 20 jours
  - Achète au cassure (break above)
  - Vend à la perte du cassure

---

## 🐛 Dépannage

### ❌ Erreur : "ModuleNotFoundError: No module named 'src'"

**Solution** :
```bash
# Assurez-vous que le terminal est dans le bon dossier
cd c:\Users\emili\Documents\trading_bot

# Vérifiez que l'environnement virtuel est activé (.venv)
.venv\Scripts\Activate.ps1  # Windows

# Réinstallez les dépendances
pip install -r requirements.txt
```

### ❌ Erreur : "Dashboard won't start" ou "Port 8050 already in use"

**Solution** :
```bash
# Trouvez et tuez le processus utilisant le port
netstat -ano | findstr :8050
taskkill /PID [PID] /F

# Ou utilisez un autre port
python -m src.dashboard.backtest_dashboard --port 8051
```

### ❌ "No data available" ou "API rate limit exceeded"

**Solution** :
- Attendre 1-2 minutes (yfinance a un rate limit)
- Réduisez le nombre de symboles testés
- Augmentez `cache_ttl_seconds` dans settings.yaml

### ❌ Tests échouent avec "NameError: name 'DCAStrategy' not found"

**Solution** :
```bash
# L'environnement a changé. Réinstallez clean :
pip install --force-reinstall -r requirements.txt
python -m pytest tests/test_core.py -q --tb=short
```

### ❌ "BROKER_API_KEY not found" en mode live

**Solution** :
1. Créez `.env` à la racine
2. Remplissez-le avec vos clés (voir section Configuration)
3. Redémarrez le bot

---

## 📚 Lectures supplémentaires

- **Checklist LIVE** : [docs/LIVE_CHECKLIST.md](docs/LIVE_CHECKLIST.md) — **OBLIGATOIRE** avant trading réel
- **Code source** : Explorez `src/strategies/` pour comprendre comment fonctionnent les stratégies
- **Tests** : Regardez `tests/test_core.py` pour des exemples d'utilisation

---

## ⚖️ Avertissements légaux

- **Pas de garantie** : Les performances passées ne garantissent pas les résultats futurs
- **Risque de perte** : Ce bot peut perdre 100% du capital investi
- **Pas de conseil financier** : Consultez un professionnel avant de trader
- **Conformité locale** : Assurez-vous que le trading algorithmique est légal dans votre juridiction
- **Impôts** : Le trading fréquent peut avoir des implications fiscales importantes

---

## 💬 Support

- **Issues** : Créez un issue sur GitHub
- **Email** : [your-email@example.com]
- **Discord** : [Lien serveur Discord]

---

**Bonne chance ! 🚀**

*Rappel : La gestion du risque est la PREMIÈRE priorité. Ne désactivez jamais le Risk Manager.*