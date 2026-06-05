# 📖 Guide d'Installation Complet (Visuel + Terminal)

*Pour quelqu'un qui n'a JAMAIS utilisé de terminal avant*

---

## 🖥️ Partie 1: Installer Python

### ✅ Étape 1.1 : Télécharger Python

1. Ouvrez votre navigateur
2. Allez sur https://www.python.org/downloads/
3. Cliquez sur le **gros bouton jaune** "Download Python 3.12.x"

### ✅ Étape 1.2 : Installer Python

**Windows** :
1. Ouvrez le fichier téléchargé (`.exe`)
2. **IMPORTANT** : Cochez la case **"Add Python to PATH"** ✓
3. Cliquez "Install Now"
4. Attendez la fin et fermez

**macOS** :
1. Ouvrez le fichier téléchargé (`.pkg`)
2. Suivez les étapes
3. Fermez quand c'est fini

**Linux (Ubuntu)** :
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

### ✅ Étape 1.3 : Vérifier l'installation

Ouvrez un nouveau terminal/PowerShell et tapez :
```bash
python --version
```

Vous devriez voir :
```
Python 3.12.x
```

Si vous voyez `command not found`, redémarrez votre ordinateur et réessayez.

---

## 🖥️ Partie 2: Installer Git

### ✅ Étape 2.1 : Télécharger Git

1. Allez sur https://git-scm.com/
2. Cliquez "Download"
3. Téléchargez la version pour votre OS

### ✅ Étape 2.2 : Installer Git

**Windows** : Double-cliquez le fichier `.exe` et suivez les étapes (gardez les valeurs par défaut)

**macOS** : Installez via Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

**Linux** :
```bash
sudo apt install git
```

### ✅ Étape 2.3 : Vérifier l'installation

```bash
git --version
```

---

## 📁 Partie 3: Cloner le Projet

### ✅ Étape 3.1 : Créer un dossier

**Windows (PowerShell)** :
```bash
mkdir C:\Users\[votre-username]\Documents\trading_bot
cd C:\Users\[votre-username]\Documents\trading_bot
```

**macOS/Linux** :
```bash
mkdir ~/trading_bot
cd ~/trading_bot
```

Remplacez `[votre-username]` par votre nom d'utilisateur Windows.

### ✅ Étape 3.2 : Cloner depuis GitHub

```bash
git clone https://github.com/[auteur]/trading_bot.git .
```

Remplacez `[auteur]` par le nom d'utilisateur GitHub de l'auteur du projet.

Vous devriez voir des fichiers téléchargés :
```
README.md
requirements.txt
config/
src/
tests/
data/
docs/
```

---

## 🐍 Partie 4: Créer l'Environnement Virtuel Python

*L'environnement virtuel isole les dépendances du projet des autres projets Python sur votre ordinateur.*

### ✅ Étape 4.1 : Créer l'environnement

Vous devez être dans le dossier `trading_bot` (vous y êtes après l'étape 3.2).

```bash
python -m venv .venv
```

Cela créera un dossier `.venv` (caché par défaut).

### ✅ Étape 4.2 : Activer l'environnement

**Windows (PowerShell)** :
```bash
.venv\Scripts\Activate.ps1
```

Si vous avez une erreur `"Cannot be loaded because running scripts is disabled"`, exécutez d'abord :
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Puis réessayez le `Activate.ps1`.

**Windows (CMD)** :
```bash
.venv\Scripts\activate.bat
```

**macOS/Linux (Bash)** :
```bash
source .venv/bin/activate
```

### ✅ Étape 4.3 : Vérifier l'activation

Vous devez voir `(.venv)` au début de votre ligne de commande :

```
(.venv) PS C:\Users\...\trading_bot>
```

ou 

```
(.venv) user@machine trading_bot %
```

---

## 📦 Partie 5: Installer les Dépendances

*Toutes les bibliothèques Python dont le bot a besoin.*

### ✅ Étape 5.1 : Mettre à jour pip

```bash
python -m pip install --upgrade pip
```

### ✅ Étape 5.2 : Installer les dépendances

```bash
pip install -r requirements.txt
```

Cela installera :
- `pandas` (données)
- `yfinance` (données de marché)
- `dash` (dashboard)
- `pytest` (tests)
- Et 10+ autres...

L'installation prend **2-5 minutes**. Attendez qu'elle se termine.

### ✅ Étape 5.3 : Vérifier l'installation

```bash
python -m pytest tests/test_core.py -q
```

Vous devriez voir :
```
39 passed
```

Si c'est bon, continuez ! ✅

---

## 🔑 Partie 6: Configurer les Secrets (Variables d'Environnement)

*Pour que le bot puisse se connecter à votre broker et autres services.*

### ✅ Étape 6.1 : Créer le fichier `.env`

**Méthode 1 (Facile - Terminal)** :
```bash
cp .env.example .env
```

**Méthode 2 (Manuel - Fichier)** :
1. Ouvrez VS Code ou Notepad
2. Créez un nouveau fichier
3. Nommez-le `.env` (attention: il n'y a pas de nom avant le `.env`)
4. Sauvegardez-le à la racine du projet (à côté de `README.md`)

### ✅ Étape 6.2 : Remplir `.env`

Ouvrez le fichier `.env` avec un éditeur de texte.

Pour le **trading papier** (test, sans argent réel), remplissez :
```env
BROKER_API_KEY=pk_test_xxxxxxxxxxxxx
BROKER_API_SECRET=yyyyyyyyyyyyyyyyyyyyyyy
BROKER_BASE_URL=https://paper-api.alpaca.markets

ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Pour le **trading réel** (argent réel), changez :
```env
BROKER_BASE_URL=https://api.alpaca.markets
```

**Où trouver vos clés** :
1. Allez sur https://app.alpaca.markets/
2. Connectez-vous avec votre compte
3. Allez sur "API Keys" dans les paramètres
4. Copiez/collez les clés dans `.env`

### ✅ Étape 6.3 : Vérifier que `.env` est ignoré

Ouvrez `.gitignore` (un fichier à la racine du projet).

Vous devriez voir `.env` dedans. Si absent, ajoutez-le :
```
.env
.DS_Store
__pycache__
.venv
```

**Ceci empêche d'accidentellement envoyer vos clés secrètes sur GitHub.**

---

## ✅ Partie 7: Première Utilisation

*Maintenant que tout est installé, testons !*

### ✅ Étape 7.1 : Exécuter un backtest

```bash
python -m src.main \
  --symbols SPY QQQ \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --strategies tactical_dca
```

Attendez **1-2 minutes**.

Résultat attendu : Un rapport avec rendement, Sharpe ratio, max drawdown, etc.

### ✅ Étape 7.2 : Lancer le dashboard

```bash
python -m src.dashboard.backtest_dashboard
```

Puis ouvrez : http://localhost:8050/

Vous devriez voir un dashboard interactif avec des graphiques.

### ✅ Étape 7.3 : Tester le mode papier

```bash
python -m src.main --mode paper
```

Le bot simule le trading en direct avec des données réelles. Vous verrez les logs en direct.

---

## 🎯 Prochaines Étapes

1. ✅ **Lisez le README.md** — Explique chaque concept
2. ✅ **Testez 5+ backtests** — Différentes stratégies, dates, symboles
3. ✅ **Lancez le trading papier** — 30+ jours minimum
4. ⚠️ **Lisez docs/LIVE_CHECKLIST.md** — OBLIGATOIRE avant argent réel
5. 💰 **Passez au live** — Seulement après tout cela

---

## 🆘 Si quelque chose ne marche pas

Voir : [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📖 Structure des fichiers

Après l'installation, votre dossier ressemble à :

```
trading_bot/
├── .venv/                 ← Environnement Python (créé à l'étape 4)
├── .env                   ← Vos secrets (créé à l'étape 6)
├── .env.example           ← Exemple de secrets (ignore this)
├── config/
│   ├── settings.yaml      ← Paramètres globaux
│   ├── risk.yaml          ← Limites de risque
│   └── strategies.yaml    ← Configuration des stratégies
├── src/                   ← Code principal
├── tests/                 ← Tests
├── data/                  ← Données et logs
├── docs/
│   ├── LIVE_CHECKLIST.md  ← ⚠️ LIS AVANT ARGENT RÉEL
│   └── TROUBLESHOOTING.md ← Aide
├── README.md              ← Documentation complète
├── QUICKSTART.md          ← Raccourci (5 min)
└── requirements.txt       ← Dépendances Python
```

---

**Félicitations ! Vous êtes prêt ! 🚀**

Lisez maintenant le [README.md](README.md) complet.
