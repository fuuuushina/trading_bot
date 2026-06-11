"""
src/analysis/sector_universe.py

Maps financial themes/sectors to ticker lists.
Used by ThemeAnalyzer and ThematicMomentumStrategy.
"""
from __future__ import annotations

SECTOR_UNIVERSE: dict[str, dict] = {
    "pharma_biotech": {
        "label": "Pharma & Biotech",
        "description": "Pharmaceutiques, biotechnologies, santé",
        "tickers": ["LLY", "ABBV", "MRK", "PFE", "JNJ", "MRNA", "AMGN", "GILD", "BMY"],
        "keywords": [
            "fda", "drug", "clinical trial", "biotech", "pharma", "medicine",
            "vaccine", "cancer", "therapy", "approval", "drug approval", "eli lilly",
            "pfizer", "johnson", "merck", "abbvie",
        ],
    },
    "ai_software": {
        "label": "IA & Logiciel",
        "description": "Intelligence artificielle, logiciels, cloud",
        "tickers": ["MSFT", "GOOGL", "META", "CRM", "PLTR", "SNOW", "NOW", "AAPL", "NFLX"],
        "keywords": [
            "ai", "artificial intelligence", "cloud", "software", "saas", "llm",
            "chatgpt", "openai", "microsoft", "google", "meta", "machine learning",
            "data center", "generative ai",
        ],
    },
    "semiconductors": {
        "label": "Semi-conducteurs",
        "description": "Semi-conducteurs, puces électroniques",
        "tickers": ["NVDA", "AMD", "AVGO", "QCOM", "AMAT", "INTC", "MU", "ASML", "ARM", "SMCI"],
        "keywords": [
            "chip", "semiconductor", "gpu", "nvidia", "amd", "wafer", "fab",
            "export control", "broadcom", "qualcomm", "intel", "memory", "hbm",
        ],
    },
    "energy": {
        "label": "Énergie",
        "description": "Énergie, pétrole, gaz naturel",
        "tickers": ["XOM", "CVX", "COP", "SLB", "OXY", "PSX", "MPC", "XLE"],
        "keywords": [
            "oil", "gas", "opec", "crude", "energy", "petroleum", "refinery",
            "barrel", "brent", "wti", "exxon", "chevron", "shell",
        ],
    },
    "fintech_banking": {
        "label": "Finance & Banque",
        "description": "Finance, banques, paiements, crypto institutionnel",
        "tickers": ["JPM", "GS", "MS", "V", "MA", "BAC", "WFC", "AXP", "COIN", "MSTR"],
        "keywords": [
            "bank", "fed", "federal reserve", "rate", "financial", "payment",
            "fintech", "lending", "credit", "jpmorgan", "goldman", "interest rate",
            "yield", "bond", "bitcoin", "crypto", "coinbase", "microstrategy",
        ],
    },
    "ev_clean_energy": {
        "label": "VE & Énergie propre",
        "description": "Véhicules électriques, énergies renouvelables",
        "tickers": ["TSLA", "ENPH", "NEE", "FSLR", "CEG", "RIVN"],
        "keywords": [
            "electric vehicle", "ev", "solar", "wind", "renewable", "battery",
            "tesla", "clean energy", "green energy", "lithium", "charging",
        ],
    },
    "defense_aerospace": {
        "label": "Défense & Aéro",
        "description": "Défense, aérospatiale, sécurité",
        "tickers": ["LMT", "RTX", "NOC", "BA", "GD", "LDOS", "HII"],
        "keywords": [
            "defense", "military", "weapon", "contract", "pentagon", "nato",
            "aerospace", "missile", "lockheed", "raytheon", "boeing", "war",
            "geopolitical", "conflict",
        ],
    },
    "retail_consumer": {
        "label": "Distribution & Conso",
        "description": "Distribution, consommation, e-commerce, global",
        "tickers": ["AMZN", "WMT", "COST", "TGT", "HD", "NKE", "MCD", "BABA"],
        "keywords": [
            "retail", "consumer", "spending", "shopping", "ecommerce", "amazon",
            "walmart", "costco", "inflation", "consumer confidence", "gdp",
        ],
    },
}


def get_tickers_for_themes(
    theme_scores: dict[str, float],
    min_score: float = 0.35,
    max_per_sector: int = 4,
    max_total: int = 16,
) -> list[str]:
    """Return tickers from sectors scoring above min_score, sorted by score descending."""
    selected: list[tuple[float, str]] = []
    for sector, score in sorted(theme_scores.items(), key=lambda x: -x[1]):
        if score < min_score:
            continue
        tickers = SECTOR_UNIVERSE.get(sector, {}).get("tickers", [])
        for t in tickers[:max_per_sector]:
            selected.append((score, t))
    result = [t for _, t in selected[:max_total]]
    return list(dict.fromkeys(result))  # deduplicate preserving order


def get_all_tickers() -> list[str]:
    """All unique tickers across all sectors."""
    seen: set[str] = set()
    result: list[str] = []
    for info in SECTOR_UNIVERSE.values():
        for t in info["tickers"]:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result
