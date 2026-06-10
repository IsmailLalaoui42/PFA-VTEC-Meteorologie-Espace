# PFA VTEC et météorologie de l'espace — code source

Auteur : Ismail LALAOUI RACHIDI
Encadrant : Prof. Mohamed KAAB
Établissement : ENSA Beni Mellal — Université Sultan Moulay Slimane

Station d'extraction : Beni Mellal, Maroc — 32.34° N, 6.36° W.
Période couverte : octobre 2015 → septembre 2025 (≈ 10 ans).

---

## Arborescence

```
.
├── 01_data_engineering_PFA.py        Pipeline d'ingestion + nettoyage OMNI/IONEX
├── 02_daily_sw_indices_PFA.py        Fichier daily : stats VTEC + indices F10.7/Kp/Dst
├── config/
│   └── config.yaml                    Source de vérité unique (chemins, station, thème)
├── pfa_vtec/                          Package Python du projet
│   ├── data/                          Loaders + schéma des CSV
│   ├── viz/                           Thème éditorial Matplotlib + Plotly
│   ├── dashboard/                     Application Dash + assets CSS
│   ├── features/                      Feature engineering
│   ├── models/                        Modules forecasting et anomaly
│   ├── evaluation/                    Walk-forward CV + métriques
│   ├── cli.py                         Entrée CLI Typer
│   ├── io_paths.py                    Résolution centralisée des chemins
│   └── __main__.py
├── output/                            Données de sortie produites par le pipeline
│   ├── OMNI_30min_cleaned.csv         Vent solaire OMNI 30-min nettoyé
│   ├── VTEC_GIM_30min_BeniMellal.csv  VTEC interpolé sur Beni Mellal
│   ├── MERGED_OMNI_VTEC_30min.csv     Fusion 30-min (socle commun du projet)
│   ├── storm_catalog_Bz_lt_minus10.csv  Catalogue d'événements Bz < −10 nT
│   └── DAILY_VTEC_SW_indices.csv      Stats journalières VTEC + F10.7 / Kp / Dst
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Données de sortie

Toutes produites par `01_data_engineering_PFA.py` à partir de deux sources publiques :

- **OMNI solar wind** (NOAA / SPDF) — résolution native 5 min, période 2015–2025.
- **IONEX GIM** (IGS) — cartes TEC globales toutes les 2 heures, ré-échantillonnées
  à 30 min et interpolées bilinéairement sur le point Beni Mellal.

| Fichier | Granularité | Lignes | Couverture VTEC |
|---|---|---|---|
| `OMNI_30min_cleaned.csv` | 30 min | 192 864 | — |
| `VTEC_GIM_30min_BeniMellal.csv` | 30 min | 87 531 | 45.4 % |
| `MERGED_OMNI_VTEC_30min.csv` | 30 min | 192 864 | 45.4 % |
| `storm_catalog_Bz_lt_minus10.csv` | événementiel | 1 125 | — |

Un fichier complémentaire, produit par `02_daily_sw_indices_PFA.py`, aligne les
statistiques journalières du VTEC avec les proxies canoniques de météo de l'espace :

| Fichier | Granularité | Lignes | Contenu |
|---|---|---|---|
| `DAILY_VTEC_SW_indices.csv` | journalier | 3 650 | Stats VTEC (moy. / méd. / σ / min / max / n) + F10.7, Kp, Ap, Dst |

Indices téléchargés depuis leurs centres de référence : **F10.7** et **Kp / Ap** (GFZ
Helmholtz Centre for Geosciences), **Dst** (WDC for Geomagnetism, Kyoto). Les
téléchargements sont mis en cache dans `data_cache/` ; une seconde exécution est
hors-ligne et les jours sans donnée restent vides (aucune valeur n'est inventée).

---

## Reproduction

### Prérequis

- Python **3.10+** (testé sur 3.11.9)
- Windows 11, Linux ou macOS

### Installation

```powershell
git clone <url-du-depot> pfa-vtec
Set-Location pfa_vtec
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Vérification rapide

```powershell
python -m pfa_vtec smoke
```

Sortie attendue : nombre de lignes, couverture VTEC, chemins de configuration.

### Lancer le tableau de bord interactif

```powershell
python -m pfa_vtec dashboard
# Ouvrir http://127.0.0.1:8050/
```

### Régénérer les figures statiques

```powershell
python -m pfa_vtec figures
```

### Réexécuter le pipeline d'ingestion (optionnel, ≈ 25 min)

```powershell
python 01_data_engineering_PFA.py
```

Reproduit l'intégralité des fichiers CSV dans `output/`.

---

## Onglets du tableau de bord

Le tableau de bord expose les deux vues de l'axe **Visualisation** :

| Onglet | Contenu |
|---|---|
| **A — VTEC Explorer** | Série temporelle interactive, KPIs, slider de plage |
| **B — Storm Catalog** | Table des événements Bz < −10 nT, statistiques 2024 |


---

## Commandes disponibles

```
python -m pfa_vtec smoke         Vérification du câblage (données + config)
python -m pfa_vtec dashboard     Démarre le tableau de bord sur :8050
python -m pfa_vtec figures       Régénère les figures statiques
python -m pfa_vtec features      Construit le cache de descripteurs
python 02_daily_sw_indices_PFA.py  Fichier daily VTEC + indices F10.7/Kp/Dst
```

---

## Licence

Code source distribué sous licence **MIT** (voir `LICENSE`).
Les données externes (OMNI NOAA/SPDF, IONEX IGS) restent soumises aux licences
de leurs producteurs respectifs.

Travail académique — ENSA Beni Mellal, Université Sultan Moulay Slimane.
