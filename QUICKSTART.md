# 🚀 Quick Start — 5 minutes

Tu viens de cloner le projet et tu veux tester rapidement ? Voici le raccourci.

## 1️⃣ Installation (1 minute)

```bash
# Cloner et entrer dans le dossier
git clone https://github.com/[username]/trading_bot.git
cd trading_bot

# Créer l'environnement
python -m venv .venv

# Activer l'environnement
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## 2️⃣ Vérifier que ça marche (30 sec)

```bash
python -m pytest tests/test_core.py -q
# Devrait afficher: "39 passed"
```

## 3️⃣ Lancer le Backtest (1 minute)

```bash
python -m src.main \
  --symbols SPY QQQ \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --strategies tactical_dca trend_following
```

**Résultat** : Rendement, Sharpe ratio, max drawdown, etc.

## 4️⃣ Lancer le Dashboard (2 minutes)

```bash
python -m src.dashboard.backtest_dashboard
```

Ouvrez : `http://localhost:8050/`

Vous devriez voir :
- Graphique de performance
- Tableau des trades
- Sliders pour ajuster les paramètres

## 5️⃣ Tester le Trading Papier

```bash
python -m src.main --mode paper
```

Le bot simule le trading en direct avec des données réelles.

---

## ✅ Prochaines étapes

- Lisez le **[README.md](README.md)** pour la version complète
- Consultez **[docs/LIVE_CHECKLIST.md](docs/LIVE_CHECKLIST.md)** avant trading réel
- Testez différentes stratégies dans `config/strategies.yaml`
- Lancez le trading papier pendant 30 jours minimum

---

## ⚠️ Important

- **NE passez PAS au mode LIVE sans lire la checklist**
- Le trading papier est GRATUIT — utilisez-le pour tester
- Les pertes sont possibles — testez d'abord sur l'historique

---

**Questions ?** Voir README.md ou docs/LIVE_CHECKLIST.md
