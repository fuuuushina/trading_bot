# 🔧 Guide de Dépannage Détaillé

## Erreur: "ModuleNotFoundError"

### Symptômes
```
ModuleNotFoundError: No module named 'src'
ModuleNotFoundError: No module named 'config'
```

### Causes possibles
1. ❌ Mauvais dossier de travail
2. ❌ Environnement virtuel non activé
3. ❌ Dépendances non installées

### Solutions

**Étape 1** : Vérifiez le dossier de travail
```bash
# Vous devez être dans trading_bot/
pwd              # macOS/Linux
cd               # Windows

# Assurez-vous de voir :
# - src/
# - config/
# - README.md
# - requirements.txt
```

**Étape 2** : Activez l'environnement virtuel
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS/Linux (Bash)
source .venv/bin/activate

# Vous devriez voir (.venv) au début de votre ligne
```

**Étape 3** : Réinstallez les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Étape 4** : Testez
```bash
python -m pytest tests/test_core.py -q
```

---

## Erreur: "Port 8050 already in use"

### Symptôme
```
OSError: [Errno 48] Address already in use
```

### Solution 1 : Tuer le processus (Windows)
```bash
# Trouvez le PID utilisant le port 8050
netstat -ano | findstr :8050

# Tuez-le
taskkill /PID [PID] /F
```

### Solution 2 : Tuer le processus (macOS/Linux)
```bash
# Trouvez le PID
lsof -i :8050

# Tuez-le
kill -9 [PID]
```

### Solution 3 : Utilisez un autre port
```bash
python -m src.dashboard.backtest_dashboard --port 8051
```

---

## Erreur: "No data available" ou "API rate limit exceeded"

### Symptôme
```
YFinance Error: No data found for symbol
HTTP 429: Too Many Requests
```

### Causes
- Trop de requêtes yfinance d'un coup
- Symbole invalide ou délisting
- Problème réseau

### Solutions

**Solution 1** : Attendez et réessayez
```bash
# Attendez 2-3 minutes
# yfinance a un rate limit d'environ 2000 req/heure
```

**Solution 2** : Testez moins de symboles
```bash
# Au lieu de:
python -m src.main --symbols SPY QQQ AAPL MSFT NVDA ...

# Testez seulement:
python -m src.main --symbols SPY QQQ
```

**Solution 3** : Augmentez le cache
```yaml
# config/settings.yaml
data:
  cache_ttl_seconds: 3600   # 1 heure au lieu de 5 min
```

**Solution 4** : Vérifiez les symboles
```bash
# Ces symboles sont valides:
SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, GLD, TLT

# Évitez les symboles délisting ou trop obscurs
```

---

## Erreur: "NameError: name 'DCAStrategy' not found"

### Symptôme
```
NameError: name 'DCAStrategy' is not defined
```

### Cause
- Ancien code qui référence `DCAStrategy`
- Les stratégies ont été renommées en `TacticalDCAStrategy` et `TrueDCAStrategy`

### Solution
```bash
# Réinstallez les dépendances
pip install --force-reinstall -r requirements.txt

# Réexécutez les tests
python -m pytest tests/test_core.py -q --tb=short
```

---

## Erreur: "BROKER_API_KEY not found" en mode live

### Symptôme
```
KeyError: 'BROKER_API_KEY'
```

### Cause
- Fichier `.env` manquant ou incomplet
- Mauvais nom de variable

### Solution

**Étape 1** : Créez `.env`
```bash
# Copiez le fichier exemple
cp .env.example .env

# OU manuellement, créez un fichier `.env` à la racine
```

**Étape 2** : Remplissez vos clés
```bash
# Ouvrez .env dans votre éditeur et remplissez:
BROKER_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
BROKER_API_SECRET=yyyyyyyyyyyyyyyyyyyyyyyy
BROKER_BASE_URL=https://paper-api.alpaca.markets

# Pour le LIVE (réel):
# BROKER_BASE_URL=https://api.alpaca.markets
```

**Étape 3** : Vérifiez que `.env` est ignoré par git
```bash
# Vérifiez .gitignore
cat .gitignore

# Vous devriez voir ".env" dedans
```

**Étape 4** : Redémarrez le bot
```bash
python -m src.main --mode live
```

---

## Erreur: "Dashboard doesn't update after code changes"

### Symptôme
- Code modifié mais dashboard affiche toujours l'ancienne version
- Graphiques ou tableaux inchangés

### Cause
- Dash/Flask met en cache le code en mémoire

### Solution

**Solution 1** : Redémarrez le serveur Dash
```bash
# Appuyez sur Ctrl+C pour arrêter
# Puis relancez:
python -m src.dashboard.backtest_dashboard
```

**Solution 2** : Forcez le rafraîchissement du navigateur
```
Navigateur -> Ctrl+Shift+R (ou Cmd+Shift+R macOS)
```

**Solution 3** : Activez le mode dev
```bash
# Dash peut auto-recharger en mode développement
export DASH_HOT_RELOAD=true    # macOS/Linux
set DASH_HOT_RELOAD=true       # Windows
python -m src.dashboard.backtest_dashboard
```

---

## Erreur: "Python not found" ou "command not found: python"

### Symptôme
```
'python' is not recognized as an internal or external command
'python' command not found
```

### Cause
- Python n'est pas installé
- Python n'est pas dans le PATH

### Solution

**Windows** :
1. Téléchargez Python depuis https://www.python.org/downloads/
2. **IMPORTANT** : Cochez "Add Python to PATH" lors de l'installation
3. Redémarrez le terminal
4. Testez :
   ```bash
   python --version
   ```

**macOS** :
```bash
# Installez Homebrew si absent
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installez Python
brew install python@3.12

# Vérifiez
python3 --version
```

**Linux (Ubuntu/Debian)** :
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

python3 --version
```

---

## Erreur: Tests échouent

### Symptôme
```
FAILED tests/test_core.py::test_xxx - AssertionError
```

### Solution

**Étape 1** : Voir les détails de l'erreur
```bash
python -m pytest tests/test_core.py -v --tb=long
```

**Étape 2** : Cas courants

**Cas A** : Données manquantes
```bash
# Réinitialisez le cache
rm -rf data/cache/
python -m pytest tests/test_core.py -q
```

**Cas B** : Dépendances incompatibles
```bash
pip install --force-reinstall -r requirements.txt
python -m pytest tests/test_core.py -q
```

**Cas C** : Timezone incorrect
```yaml
# config/settings.yaml
system:
  timezone: "UTC"  # ou votre timezone
```

---

## Erreur: "Dashboard hangs" ou "freezes"

### Symptôme
- Dashboard charge pendant longtemps
- Interface ne répond pas
- Crash après quelques clics

### Causes
- Trop de données à traiter
- Requête API lente

### Solution

**Solution 1** : Testez sur moins de symboles
```bash
# Réduisez le nombre de symboles dans les paramètres du dashboard
```

**Solution 2** : Augmentez le timeout
```yaml
# config/settings.yaml
data:
  cache_ttl_seconds: 3600
```

**Solution 3** : Relancez le dashboard
```bash
# Appuyez sur Ctrl+C
python -m src.dashboard.backtest_dashboard
```

---

## Erreur: "Max retries exceeded" (réseau)

### Symptôme
```
MaxRetryError: HTTPSConnectionPool(host='api.example.com')
```

### Cause
- Pas de connexion internet
- Pare-feu bloque les requêtes
- Serveur indisponible

### Solution

**Solution 1** : Vérifiez la connexion
```bash
# Testez une requête simple
python -c "import yfinance as yf; print(yf.Ticker('SPY').info)"
```

**Solution 2** : Vérifiez le pare-feu
- Désactivez temporairement le pare-feu
- Vérifiez les règles antivirus

**Solution 3** : Attendez
- Si le serveur est down, attendez et réessayez

---

## Erreur: "Memory leak" ou "Out of memory"

### Symptôme
- Python utilise de plus en plus de RAM
- Le bot se ferme après quelques heures

### Causes
- Boucle infiniment accumule des données
- Cache mal géré

### Solution

```bash
# Videz le cache
rm -rf data/cache/

# Relancez le bot
python -m src.main --mode paper
```

---

## Erreur: "Kill switch activated unexpectedly"

### Symptôme
- Bot s'arrête sans raison
- Alerte : "Kill switch triggered"

### Causes
- Pertes dépassent le seuil
- Trop d'erreurs API
- Données manquantes

### Solution

Consultez les logs :
```bash
# Vérifiez les logs récents
tail -50 data/logs/bot_*.jsonl

# Cherchez les champs "warning" ou "error"
```

Puis ajustez les limites dans `config/risk.yaml` :
```yaml
kill_switch:
  daily_loss_pct: 0.01    # ← Augmentez si trop sensible
  max_api_errors_per_hour: 10
```

---

## Je n'ai pas trouvé ma réponse

1. Vérifiez le [README.md](README.md)
2. Consultez les [tests](tests/test_core.py) pour des exemples
3. Créez une issue sur GitHub
4. Contactez le support

---

**Bonne chance ! 🚀**
