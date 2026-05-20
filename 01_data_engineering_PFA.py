"""
╔══════════════════════════════════════════════════════════════════════════╗
║      PFA VTEC & MÉTÉOROLOGIE DE L'ESPACE — AXE VISUALISATION             ║
║      Pipeline de Data Engineering — Phase Commune                        ║
║      Point d'extraction : Beni Mellal (32.34°N, 6.36°W)                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Ce script effectue, dans l'ordre :
  1. Chargement & parsing OMNI (5 min, 2015–2025)
  2. Remplacement des sentinelles → NaN
  3. EDA complet (statistiques, taux de couverture, anomalies)
  4. Détection & gestion des valeurs aberrantes (IQR)
  5. Interpolation des lacunes courtes
  6. Agrégation 30 min (mean/std/min/max/count)
  7. Extraction VTEC depuis les fichiers IONEX disponibles
  8. Agrégation VTEC 30 min + merge OMNI
  9. Production de 9 figures EDA professionnelles
"""

import re
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
OMNI_RAW   = Path("OMNI_DATA_2015_2025_5min_resolution.txt")
IONEX_DIR  = Path("D:\pfa\ionex_dec")
OUT_DIR    = Path("D:\pfa\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Point géographique : Beni Mellal, Maroc
STA_LAT, STA_LON = 32.34, -6.36

# Sentinelles OMNI (valeur → NaN)
SENTINELS = {"Bz": 9999.99, "Vsw": 99999.9, "Pdyn": 999.99, "Np": 99.99}

# Seuils physiques (garde-fous, hors sentinelles)
PHYS_BOUNDS = {
    "Bz":   (-100.0,  100.0),   # nT  — tempêtes extrêmes : ±80 nT
    "Vsw":  ( 200.0, 1500.0),   # km/s
    "Pdyn": (   0.0,   50.0),   # nPa
    "Np":   (   0.0,  100.0),   # cm⁻³
}

# Fenêtre d'interpolation : lacunes ≤ 4 pas (20 min)
INTERP_LIMIT = 4

# ─────────────────────────────────────────────
# STYLE MATPLOTLIB
# ─────────────────────────────────────────────
PALETTE = {
    "Bz":   "#3A7BD5",   # bleu
    "Vsw":  "#E05F00",   # orange
    "Pdyn": "#1BAA75",   # vert
    "Np":   "#9B59B6",   # violet
    "VTEC": "#E74C3C",   # rouge
}
BG   = "#0F1117"
AXES = "#1A1E2B"
GRID = "#2E3450"
TEXT = "#EAECF5"

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    AXES,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.alpha":        0.6,
        "legend.facecolor":  AXES,
        "legend.edgecolor":  GRID,
        "figure.dpi":        120,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    9.5,
    })

set_dark_style()

UNITS = {"Bz": "nT", "Vsw": "km/s", "Pdyn": "nPa", "Np": "cm⁻³", "VTEC": "TECU"}
VARS  = ["Bz", "Vsw", "Pdyn", "Np"]

# ═══════════════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT & PARSING OMNI
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 1 — Chargement OMNI (5 min, 2015–2025)")
print("═"*65)

cols = ["year", "doy", "hour", "minute", "Bz", "Vsw", "Pdyn", "Np"]
df_raw = pd.read_csv(
    OMNI_RAW, sep=r"\s+", header=None, names=cols,
    dtype={"year": int, "doy": int, "hour": int, "minute": int,
           "Bz": float, "Vsw": float, "Pdyn": float, "Np": float}
)
print(f"  ✓ Lignes brutes lues : {len(df_raw):,}")

# Construction de l'index temporel UTC
df_raw["time"] = pd.to_datetime(
    df_raw["year"].astype(str) + df_raw["doy"].astype(str).str.zfill(3),
    format="%Y%j", utc=True
) + pd.to_timedelta(df_raw["hour"], unit="h") + pd.to_timedelta(df_raw["minute"], unit="m")

df_raw.set_index("time", inplace=True)
df_raw.drop(columns=["year","doy","hour","minute"], inplace=True)
df_raw.sort_index(inplace=True)

# Suppression des doublons temporels (si existants)
n_dup = df_raw.index.duplicated().sum()
df_raw = df_raw[~df_raw.index.duplicated(keep="first")]
print(f"  ✓ Doublons temporels supprimés : {n_dup}")
print(f"  ✓ Période : {df_raw.index[0]} → {df_raw.index[-1]}")

# ═══════════════════════════════════════════════════
# ÉTAPE 2 — REMPLACEMENT SENTINELLES
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 2 — Remplacement des sentinelles (→ NaN)")
print("═"*65)

df = df_raw.copy()
sentinel_counts_raw = {}
for var, sentinel in SENTINELS.items():
    n = (df[var] == sentinel).sum()
    sentinel_counts_raw[var] = n
    df.loc[df[var] == sentinel, var] = np.nan
    print(f"  {var:6s} : {n:7,} sentinelles remplacées ({100*n/len(df):.2f}%)")

# ═══════════════════════════════════════════════════
# ÉTAPE 3 — EDA PRÉLIMINAIRE
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 3 — EDA préliminaire")
print("═"*65)

stats_before = df.describe(percentiles=[.05,.25,.5,.75,.95]).T
stats_before["missing_%"] = df.isnull().mean() * 100
print("\n  Statistiques descriptives (après remplacement sentinelles) :\n")
print(stats_before[["count","mean","std","min","5%","50%","95%","max","missing_%"]].to_string())

# ═══════════════════════════════════════════════════
# ÉTAPE 4 — DÉTECTION & NETTOYAGE DES ABERRANTS
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 4 — Détection des valeurs aberrantes (IQR + bornes physiques)")
print("═"*65)

outlier_report = {}
df_clean = df.copy()

for var in VARS:
    series = df_clean[var].dropna()
    
    # Méthode IQR robuste
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR     = Q3 - Q1
    k       = 3.0   # facteur conservateur (3×IQR au lieu de 1.5)
    low_iqr  = Q1 - k * IQR
    high_iqr = Q3 + k * IQR
    
    # Bornes physiques (garde-fous absolus)
    low_phys,  high_phys  = PHYS_BOUNDS[var]
    
    # Intersection la plus restrictive
    low_final  = max(low_iqr,  low_phys)
    high_final = min(high_iqr, high_phys)
    
    mask_out = (
        (df_clean[var] < low_final) | (df_clean[var] > high_final)
    ) & df_clean[var].notna()
    
    n_out = mask_out.sum()
    df_clean.loc[mask_out, var] = np.nan
    
    outlier_report[var] = {
        "Q1": round(Q1,3), "Q3": round(Q3,3), "IQR": round(IQR,3),
        "low_iqr": round(low_iqr,3), "high_iqr": round(high_iqr,3),
        "low_final": round(low_final,3), "high_final": round(high_final,3),
        "n_outliers": n_out, "pct_outliers": round(100*n_out/len(df_clean),4)
    }
    print(f"  {var:6s} | bornes [{low_final:.2f}, {high_final:.2f}] | "
          f"aberrants → NaN : {n_out:,} ({100*n_out/len(df_clean):.4f}%)")

# ═══════════════════════════════════════════════════
# ÉTAPE 5 — INTERPOLATION DES LACUNES COURTES
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print(f"  ÉTAPE 5 — Interpolation linéaire (lacunes ≤ {INTERP_LIMIT} pas = {INTERP_LIMIT*5} min)")
print("═"*65)

df_interp = df_clean.copy()
for var in VARS:
    n_nan_before = df_interp[var].isna().sum()
    df_interp[var] = df_interp[var].interpolate(
        method="time", limit=INTERP_LIMIT, limit_direction="forward"
    )
    n_nan_after = df_interp[var].isna().sum()
    n_filled = n_nan_before - n_nan_after
    print(f"  {var:6s} : {n_filled:,} NaN interpolés | NaN résiduels : {n_nan_after:,} "
          f"({100*n_nan_after/len(df_interp):.2f}%)")

stats_after = df_interp.describe(percentiles=[.05,.25,.5,.75,.95]).T
stats_after["missing_%"] = df_interp.isnull().mean() * 100

# ═══════════════════════════════════════════════════
# ÉTAPE 6 — AGRÉGATION 30 MIN (OMNI)
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 6 — Agrégation OMNI → 30 min")
print("═"*65)

omni_30 = df_interp.resample("30min").agg(
    Bz_mean=("Bz","mean"), Bz_std=("Bz","std"), Bz_min=("Bz","min"), Bz_max=("Bz","max"),
    Vsw_mean=("Vsw","mean"), Vsw_std=("Vsw","std"),
    Pdyn_mean=("Pdyn","mean"), Pdyn_std=("Pdyn","std"),
    Np_mean=("Np","mean"),  Np_std=("Np","std"),
    n_valid=("Bz","count"),
)
# Couverture réelle par bin (6 mesures 5-min dans un bin 30-min)
omni_30["coverage_pct"] = (omni_30["n_valid"] / 6 * 100).clip(0, 100)

# Epsilon de couplage solaire: B_s (composante sud uniquement)
omni_30["Bs"] = (-omni_30["Bz_mean"]).clip(lower=0)

# Paramètre de couplage Newell (simplifié) : ε ∝ |Vsw|^(4/3) * Bs^(2/3)
omni_30["epsilon_coupling"] = (
    omni_30["Vsw_mean"].abs() ** (4/3) * omni_30["Bs"] ** (2/3)
).fillna(0)

omni_30.to_csv(OUT_DIR / "OMNI_30min_cleaned.csv")
print(f"  ✓ Sauvegardé : OMNI_30min_cleaned.csv | {len(omni_30):,} bins")


# ═══════════════════════════════════════════════════
# ÉTAPE 7 — EXTRACTION VTEC DEPUIS IONEX
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 7 — Extraction VTEC depuis les IONEX disponibles")
print("═"*65)

def _open_text(p: Path) -> io.StringIO:
    return io.StringIO(p.read_text(encoding="ascii", errors="ignore"))

def ionex_first_epoch_date(path: Path):
    try:
        f = _open_text(path)
    except Exception:
        return None
    for _ in range(400):
        line = f.readline()
        if not line: break
        if "EPOCH OF FIRST MAP" in line:
            parts = line[:60].split()[:6]
            if len(parts) >= 3:
                try:
                    yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                    return date(yr, mo, dy)
                except: pass
        if "END OF HEADER" in line: break
    return None

def read_ionex_vtec(path: Path, lat: float, lon: float) -> pd.Series:
    """Retourne une pd.Series de VTEC interpolé au point (lat, lon)."""
    f = _open_text(path)
    exp = -1
    lat1 = lat2 = dlat = lon1 = lon2 = dlon = None
    
    for _ in range(500):
        line = f.readline()
        if not line: break
        if "EXPONENT" in line:
            s = line[:8].strip()
            try: exp = int(s)
            except: pass
        if "LAT1 / LAT2 / DLAT" in line:
            try: lat1, lat2, dlat = map(float, line[:60].split()[:3])
            except: pass
        if "LON1 / LON2 / DLON" in line:
            try: lon1, lon2, dlon = map(float, line[:60].split()[:3])
            except: pass
        if "END OF HEADER" in line: break
    
    if None in (lat1, lat2, dlat, lon1, lon2, dlon):
        return pd.Series(dtype=float)
    
    nlat = int(round(abs(lat2 - lat1) / abs(dlat))) + 1
    nlon = int(round(abs(lon2 - lon1) / abs(dlon))) + 1
    lats = np.linspace(lat1, lat2, nlat)
    lons = np.linspace(lon1, lon2, nlon)
    
    # Normalise longitudes
    if lons.min() >= 0 and lons.max() > 180:
        lons = ((lons + 180) % 360) - 180
        order = np.argsort(lons)
        lons  = lons[order]
        lon_reorder = order
    else:
        lon_reorder = None
    
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        lat_flip = True
    else:
        lat_flip = False
    
    times, vals = [], []
    
    while True:
        line = f.readline()
        if not line: break
        if "START OF TEC MAP" not in line: continue
        
        # Epoch
        ep_line = None
        for _ in range(10):
            l = f.readline()
            if not l: break
            if "EPOCH OF CURRENT MAP" in l:
                ep_line = l; break
        if ep_line is None: continue
        try:
            yr,mo,dy,hh,mm,ss = map(int, ep_line[:60].split()[:6])
            t = pd.Timestamp(datetime(yr,mo,dy,hh,mm,ss,tzinfo=timezone.utc))
        except: continue
        
        tec_grid = np.full((nlat, nlon), np.nan)
        bad = False
        for ilat in range(nlat):
            hdr = f.readline()
            if not hdr or "LAT/LON1/LON2/DLON/H" not in hdr:
                bad = True; break
            row_vals = []
            while len(row_vals) < nlon:
                data = f.readline()
                if not data: bad = True; break
                chunks = [data[i:i+5] for i in range(0, len(data.rstrip("\n")), 5)]
                for c in chunks:
                    c = c.strip()
                    if c in ("", "9999"):
                        row_vals.append(np.nan)
                    else:
                        try: row_vals.append(float(c) * 10.0**exp)
                        except: row_vals.append(np.nan)
                if bad: break
            if bad: break
            tec_grid[ilat, :] = row_vals[:nlon]
        
        if bad: continue
        
        if lat_flip:   tec_grid = tec_grid[::-1, :]
        if lon_reorder is not None: tec_grid = tec_grid[:, lon_reorder]
        
        # Interpolation bilinéaire
        i = int(np.clip(np.searchsorted(lats, lat) - 1, 0, nlat-2))
        j = int(np.clip(np.searchsorted(lons, lon) - 1, 0, nlon-2))
        y1, y2 = lats[i], lats[i+1]
        x1, x2 = lons[j], lons[j+1]
        if (x2-x1)==0 or (y2-y1)==0:
            vtec_val = float(tec_grid[i,j])
        else:
            wx = (lon - x1) / (x2 - x1)
            wy = (lat - y1) / (y2 - y1)
            vtec_val = float(
                (1-wx)*(1-wy)*tec_grid[i,j] + wx*(1-wy)*tec_grid[i,j+1]
                + (1-wx)*wy*tec_grid[i+1,j] + wx*wy*tec_grid[i+1,j+1]
            )
        times.append(t)
        vals.append(vtec_val)
    
    if not times:
        return pd.Series(dtype=float)
    
    ser = pd.Series(vals, index=pd.DatetimeIndex(times).tz_convert("UTC"), name="vtec")
    ser = ser[~ser.index.duplicated(keep="first")].sort_index()
    return ser

# Mapping jour → fichier IONEX
def pick_ionex_for_day(d: date) -> Path | None:
    yy  = f"{d.year % 100:02d}"
    doy = f"{int(pd.Timestamp(d).strftime('%j')):03d}"
    yr  = d.year
    
    # Ancien format
    for name in (f"codg{doy}0.{yy}i", f"CODG{doy}0.{yy}I"):
        p = IONEX_DIR / name
        if p.exists(): return p
    
    # Nouveau format
    for pat in [
        f"COD0OPSFIN_{yr}{doy}*_GIM.INX",
        f"COD0OPSRAP_{yr}{doy}*_GIM.INX",
    ]:
        matches = list(IONEX_DIR.glob(pat))
        if matches: return matches[0]
    return None

# Parcours de tous les jours disponibles
all_vtec = []
total_days = (date(2025, 12, 31) - date(2015, 1, 1)).days + 1
processed, missing = 0, 0

d = date(2015, 1, 1)
while d <= date(2025, 12, 31):
    p = pick_ionex_for_day(d)
    if p is not None:
        try:
            ser = read_ionex_vtec(p, STA_LAT, STA_LON)
            if len(ser) > 0:
                all_vtec.append(ser)
                processed += 1
        except Exception as e:
            pass
    else:
        missing += 1
    d += timedelta(days=1)

print(f"  ✓ Jours avec IONEX : {processed:,} | Sans IONEX : {missing:,}")

if all_vtec:
    vtec_raw = pd.concat(all_vtec).sort_index()
    vtec_raw = vtec_raw[~vtec_raw.index.duplicated(keep="first")]
    
    # Nettoyage VTEC (bornes physiques)
    vtec_raw = vtec_raw.clip(0, 200)  # >200 TECU physiquement impossible
    
    # Agrégation 30 min (Series → DataFrame)
    vtec_df = vtec_raw.rename("vtec").to_frame()
    vtec_30 = vtec_df.resample("30min").agg(
        vtec_mean=("vtec","mean"), vtec_std=("vtec","std"),
        vtec_min=("vtec","min"),   vtec_max=("vtec","max"),
        n_maps=("vtec","count")
    )
    vtec_30.to_csv(OUT_DIR / "VTEC_GIM_30min_BeniMellal.csv")
    print(f"  ✓ Sauvegardé : VTEC_GIM_30min_BeniMellal.csv | {len(vtec_30):,} bins")
else:
    print("  ⚠ Aucune donnée VTEC extraite.")
    vtec_30 = pd.DataFrame()

# ═══════════════════════════════════════════════════
# ÉTAPE 8 — FUSION OMNI + VTEC → DATASET FUSIONNÉ
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 8 — Fusion OMNI 30 min + VTEC 30 min")
print("═"*65)

if not vtec_30.empty:
    merged = omni_30.join(vtec_30, how="left")
    merged.to_csv(OUT_DIR / "MERGED_OMNI_VTEC_30min.csv")
    coverage_vtec = merged["vtec_mean"].notna().mean() * 100
    print(f"  ✓ Dataset fusionné : {len(merged):,} bins | Couverture VTEC : {coverage_vtec:.1f}%")
else:
    merged = omni_30.copy()
    print("  ⚠ Fusion sans VTEC (non disponible).")

# ═══════════════════════════════════════════════════
# ÉTAPE 9 — FIGURES EDA
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  ÉTAPE 9 — Production des figures EDA")
print("═"*65)

# ───────────────────────────────────────────────────
# FIGURE 1 : Carte de disponibilité des données
# ───────────────────────────────────────────────────
print("  → Figure 1 : Carte de disponibilité (data coverage heatmap)")

# Calculer coverage mensuel par variable (données 5-min)
df_interp_reset = df_interp.copy()
df_interp_reset["year"]  = df_interp_reset.index.year
df_interp_reset["month"] = df_interp_reset.index.month

fig, axes = plt.subplots(2, 2, figsize=(14, 7))
fig.suptitle("Carte de disponibilité des données OMNI (2015–2025)\n"
             "% de mesures valides par mois", fontsize=13, fontweight="bold", y=1.01)

for ax, var in zip(axes.flat, VARS):
    pivot = df_interp_reset.groupby(["year","month"])[var].apply(
        lambda s: 100 * s.notna().mean()
    ).unstack(level="month")
    pivot = pivot.reindex(columns=range(1,13))
    
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_title(f"{var}  [{UNITS[var]}]", color=PALETTE[var], fontweight="bold")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    
    # Valeurs dans les cellules
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i,j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                       fontsize=6, color="black" if val > 50 else "white")
    
    plt.colorbar(im, ax=ax, shrink=0.8, label="%")
    ax.set_xlabel("Mois"); ax.set_ylabel("Année")

plt.tight_layout()
fig.savefig(OUT_DIR / "Fig1_data_availability_heatmap.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig1_data_availability_heatmap.png")


# ───────────────────────────────────────────────────
# FIGURE 2 : Série temporelle vue d'ensemble (annuelle)
# ───────────────────────────────────────────────────
print("  → Figure 2 : Série temporelle d'ensemble 2015–2025 (mensuel médian)")

# Moyenne mensuelle pour l'aperçu (moins lourd)
df_monthly = df_interp.resample("ME").median()

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Évolution temporelle des paramètres du vent solaire (médiane mensuelle)\n"
             "OMNI – Beni Mellal – 2015–2025", fontsize=13, fontweight="bold")

ax_labels = [
    ("Bz",   "IMF Bz (nT)",           "Champ magnétique interplanétaire Nord-Sud"),
    ("Vsw",  "V_sw (km/s)",           "Vitesse du vent solaire"),
    ("Pdyn", "P_dyn (nPa)",           "Pression dynamique"),
    ("Np",   "N_p (cm⁻³)",           "Densité protonique"),
]

for ax, (var, ylabel, desc) in zip(axes, ax_labels):
    col = PALETTE[var]
    ax.fill_between(df_monthly.index, df_monthly[var], alpha=0.25, color=col)
    ax.plot(df_monthly.index, df_monthly[var], lw=1.4, color=col, label=desc)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="x", alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Annoter les maxima notables
    idx_max = df_monthly[var].idxmax()
    ax.axvline(idx_max, color=col, lw=0.8, ls="--", alpha=0.5)
    ax.annotate(f"max\n{df_monthly[var].max():.1f}",
                xy=(idx_max, df_monthly[var].max()),
                fontsize=7, color=col,
                xytext=(10, -15), textcoords="offset points")

axes[-1].set_xlabel("Date (UTC)")
plt.tight_layout()
fig.savefig(OUT_DIR / "Fig2_timeseries_overview.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig2_timeseries_overview.png")


# ───────────────────────────────────────────────────
# FIGURE 3 : Distributions avant/après nettoyage
# ───────────────────────────────────────────────────
print("  → Figure 3 : Distributions avant/après nettoyage (histogrammes + KDE)")

fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle("Distributions des variables OMNI — Avant vs Après nettoyage\n"
             "Bleu foncé = brut (sans sentinelles) | Couleur = nettoyé+interpolé",
             fontsize=12, fontweight="bold")

for col_i, var in enumerate(VARS):
    col = PALETTE[var]
    
    # Avant nettoyage (df = après sentinelles, avant outlier removal)
    ax_top = axes[0, col_i]
    raw_vals = df[var].dropna()
    ax_top.hist(raw_vals, bins=120, color="#3A5A8A", alpha=0.7, density=True, label="Brut")
    ax_top.set_title(f"{var} [{UNITS[var]}]\nAvant nettoyage", fontsize=9)
    ax_top.set_xlabel(UNITS[var], fontsize=8)
    if col_i == 0: ax_top.set_ylabel("Densité", fontsize=8)
    ax_top.legend(fontsize=7)
    
    # Après nettoyage
    ax_bot = axes[1, col_i]
    clean_vals = df_interp[var].dropna()
    ax_bot.hist(clean_vals, bins=120, color=col, alpha=0.75, density=True, label="Nettoyé")
    
    # KDE overlay
    from scipy.stats import gaussian_kde
    if len(clean_vals) > 100:
        x_kde = np.linspace(clean_vals.min(), clean_vals.max(), 300)
        kde = gaussian_kde(clean_vals.sample(min(50000, len(clean_vals)), random_state=42))
        ax_bot.plot(x_kde, kde(x_kde), color="white", lw=1.5, alpha=0.9, label="KDE")
    
    # Lignes statistiques
    for pct, ls in [(0.5,"--"),(0.05,":"),(0.95,":")]:
        val = clean_vals.quantile(pct)
        ax_bot.axvline(val, color="yellow", lw=0.9, ls=ls, alpha=0.7)
    
    ax_bot.set_title(f"{var} [{UNITS[var]}]\nAprès nettoyage + interpolation", fontsize=9)
    ax_bot.set_xlabel(UNITS[var], fontsize=8)
    if col_i == 0: ax_bot.set_ylabel("Densité", fontsize=8)
    ax_bot.legend(fontsize=7)

plt.tight_layout()
fig.savefig(OUT_DIR / "Fig3_distributions_before_after.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig3_distributions_before_after.png")


# ───────────────────────────────────────────────────
# FIGURE 4 : Box-plots par année (tendances)
# ───────────────────────────────────────────────────
print("  → Figure 4 : Box-plots par année")

df_interp_y = df_interp.copy()
df_interp_y["year"] = df_interp_y.index.year

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Variabilité interannuelle des paramètres du vent solaire\n"
             "Box-plots par année (données 5 min nettoyées)", fontsize=12, fontweight="bold")

for ax, var in zip(axes.flat, VARS):
    col = PALETTE[var]
    data_by_year = [df_interp_y.loc[df_interp_y["year"]==yr, var].dropna().values
                    for yr in range(2015, 2026)]
    
    bp = ax.boxplot(data_by_year, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color=col, alpha=0.8),
                    capprops=dict(color=col, alpha=0.8),
                    flierprops=dict(marker=".", ms=1, alpha=0.2))
    
    for i, patch in enumerate(bp["boxes"]):
        alpha_val = 0.4 + 0.06 * i
        patch.set_facecolor(col)
        patch.set_alpha(min(alpha_val, 0.9))
    
    ax.set_title(f"{var}  [{UNITS[var]}]", color=col, fontweight="bold")
    ax.set_xticklabels(range(2015, 2026), rotation=45, fontsize=8)
    ax.set_xlabel("Année")
    ax.set_ylabel(UNITS[var])
    ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
fig.savefig(OUT_DIR / "Fig4_boxplots_per_year.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig4_boxplots_per_year.png")


# ───────────────────────────────────────────────────
# FIGURE 5 : Matrice de corrélation + scatter 30 min
# ───────────────────────────────────────────────────
print("  → Figure 5 : Matrice de corrélation (Pearson & Spearman)")

sample_30 = omni_30[["Bz_mean","Vsw_mean","Pdyn_mean","Np_mean"]].dropna()
sample_30.columns = ["Bz","Vsw","Pdyn","Np"]

fig = plt.figure(figsize=(13, 10))
fig.suptitle("Matrice de corrélation & scatter — OMNI 30 min (2015–2025)\n"
             "Triangle inf. : Pearson | Triangle sup. : Spearman", fontsize=12, fontweight="bold")

n = 4
gs = gridspec.GridSpec(n, n, figure=fig, hspace=0.08, wspace=0.08)

for i, var_i in enumerate(VARS):
    for j, var_j in enumerate(VARS):
        ax = fig.add_subplot(gs[i, j])
        
        if i == j:
            # Diagonale : distribution
            vals = sample_30[var_i].dropna()
            ax.hist(vals, bins=60, color=PALETTE[var_i], alpha=0.7, density=True)
            ax.set_facecolor(AXES)
        elif i > j:
            # Triangle inf : scatter + Pearson
            xi = sample_30[[var_j, var_i]].dropna()
            samp = xi.sample(min(5000, len(xi)), random_state=0)
            ax.scatter(samp[var_j], samp[var_i], alpha=0.08, s=1, color=PALETTE[var_i])
            r = xi.corr().iloc[0,1]
            ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                   fontsize=8, color="white", va="top",
                   bbox=dict(boxstyle="round", fc=AXES, alpha=0.7))
            ax.set_facecolor(AXES)
        else:
            # Triangle sup : Spearman
            xi = sample_30[[var_j, var_i]].dropna()
            rs = xi.corr(method="spearman").iloc[0,1]
            abs_rs = abs(rs)
            ax.set_facecolor(mcolors.to_rgba(PALETTE[var_i], 0.15 + 0.6*abs_rs))
            ax.text(0.5, 0.5, f"ρ = {rs:.3f}", transform=ax.transAxes,
                   ha="center", va="center", fontsize=11, fontweight="bold",
                   color="white" if abs_rs > 0.4 else TEXT)
        
        # Labels bords
        if i == 0: ax.set_title(var_j, fontsize=9, color=PALETTE[var_j])
        if j == 0: ax.set_ylabel(var_i, fontsize=9, color=PALETTE[var_i])
        ax.tick_params(labelsize=6)
        if i < n-1: ax.set_xticklabels([])
        if j > 0:   ax.set_yticklabels([])

plt.savefig(OUT_DIR / "Fig5_correlation_matrix.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig5_correlation_matrix.png")


# ───────────────────────────────────────────────────
# FIGURE 6 : Climatologie mensuelle (cycle saisonnier)
# ───────────────────────────────────────────────────
print("  → Figure 6 : Climatologie mensuelle (médiane ± IQR)")

df_interp_m = df_interp.copy()
df_interp_m["month"] = df_interp_m.index.month

MONTH_LABELS = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Climatologie mensuelle des paramètres du vent solaire\n"
             "Médiane ± IQR (2015–2025)", fontsize=12, fontweight="bold")

for ax, var in zip(axes.flat, VARS):
    col = PALETTE[var]
    grp = df_interp_m.groupby("month")[var]
    med  = grp.median()
    q25  = grp.quantile(0.25)
    q75  = grp.quantile(0.75)
    q05  = grp.quantile(0.05)
    q95  = grp.quantile(0.95)
    
    x = np.arange(1, 13)
    ax.fill_between(x, q05, q95, alpha=0.18, color=col, label="5–95%")
    ax.fill_between(x, q25, q75, alpha=0.40, color=col, label="IQR")
    ax.plot(x, med, lw=2.2, color=col, marker="o", ms=5, label="Médiane")
    
    ax.set_title(f"{var}  [{UNITS[var]}]", color=col, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xlabel("Mois")
    ax.set_ylabel(UNITS[var])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.35)

plt.tight_layout()
fig.savefig(OUT_DIR / "Fig6_monthly_climatology.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig6_monthly_climatology.png")


# ───────────────────────────────────────────────────
# FIGURE 7 : Détection de tempêtes géomagnétiques
# ───────────────────────────────────────────────────
print("  → Figure 7 : Catalogue de tempêtes (Bz < -10 nT, fenêtre 5 min)")

BZ_THRESH = -10.0  # nT — seuil tempête modérée

bz_series = df_interp["Bz"]
storms = bz_series[bz_series < BZ_THRESH]

# Agrégation par événement : séquences consécutives
storm_events = []
in_event = False
ev_start, ev_min = None, 0.0

# Downsampled loop pour performance
bz_arr = bz_series.values
t_arr  = bz_series.index

for k in range(len(bz_arr)):
    v = bz_arr[k]
    if not np.isnan(v) and v < BZ_THRESH:
        if not in_event:
            in_event = True
            ev_start = t_arr[k]
            ev_min   = v
        else:
            ev_min = min(ev_min, v)
    else:
        if in_event:
            in_event = False
            storm_events.append({"start": ev_start, "end": t_arr[k-1],
                                  "Bz_min": ev_min})
            ev_start, ev_min = None, 0.0

df_storms = pd.DataFrame(storm_events)
df_storms["duration_h"] = (df_storms["end"] - df_storms["start"]).dt.total_seconds() / 3600
df_storms.to_csv(OUT_DIR / "storm_catalog_Bz_lt_minus10.csv", index=False)

# Top 10 tempêtes
top10 = df_storms.nsmallest(10, "Bz_min")

fig, axes = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[3,1.5])
fig.suptitle(f"Détection de tempêtes géomagnétiques — IMF Bz (5 min, 2015–2025)\n"
             f"Critère : Bz < {BZ_THRESH} nT | {len(df_storms)} événements détectés",
             fontsize=12, fontweight="bold")

# Vue annuelle (médiane 3-horaire pour lisibilité)
bz_3h = bz_series.resample("3h").median()
ax0 = axes[0]
ax0.plot(bz_3h.index, bz_3h.values, lw=0.6, color=PALETTE["Bz"], alpha=0.85)
ax0.axhline(BZ_THRESH, color="red", lw=1.2, ls="--", label=f"Seuil {BZ_THRESH} nT")
ax0.axhline(0, color="white", lw=0.5, alpha=0.3)

# Surligner les top 5 tempêtes
for _, ev in top10.head(5).iterrows():
    ax0.axvspan(ev["start"], ev["end"], color="red", alpha=0.15)
    ax0.annotate(f"{ev['Bz_min']:.0f}nT",
                 xy=(ev["start"], ev["Bz_min"]),
                 fontsize=7, color="red", alpha=0.9,
                 xytext=(0, -20), textcoords="offset points")

ax0.set_ylabel("Bz (nT)")
ax0.legend(fontsize=9)
ax0.grid(alpha=0.3)

# Histogramme des durées de tempêtes
ax1 = axes[1]
dur_vals = df_storms["duration_h"].clip(upper=72)
ax1.hist(dur_vals, bins=50, color=PALETTE["Bz"], alpha=0.75, edgecolor=GRID)
ax1.set_xlabel("Durée de l'événement (heures)")
ax1.set_ylabel("Nombre d'événements")
ax1.set_title(f"Distribution des durées — médiane : {dur_vals.median():.1f} h | "
              f"max : {df_storms['duration_h'].max():.1f} h", fontsize=9)
ax1.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "Fig7_storm_detection.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"    ✓ Fig7_storm_detection.png  ({len(df_storms)} événements | "
      f"min Bz = {df_storms['Bz_min'].min():.1f} nT)")


# ───────────────────────────────────────────────────
# FIGURE 8 : VTEC + Couplage solaire
# ───────────────────────────────────────────────────
print("  → Figure 8 : VTEC observé + paramètre de couplage")

if not vtec_30.empty and "vtec_mean" in merged.columns:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("VTEC (Beni Mellal) & Paramètres de couplage ionosphère–magnétosphère\n"
                 "Résolution 30 min — GIM/IONEX + OMNI", fontsize=12, fontweight="bold")
    
    # VTEC mensuel médian
    vtec_monthly = merged["vtec_mean"].resample("ME").median()
    ax0 = axes[0]
    ax0.fill_between(vtec_monthly.index, vtec_monthly, alpha=0.3, color=PALETTE["VTEC"])
    ax0.plot(vtec_monthly.index, vtec_monthly, lw=1.5, color=PALETTE["VTEC"])
    ax0.set_ylabel("VTEC (TECU)")
    ax0.set_title("VTEC médian mensuel — GIM (Beni Mellal, 32.34°N, 6.36°W)", fontsize=10)
    ax0.grid(alpha=0.35)
    
    # Bz et Bs
    bz_m = merged["Bz_mean"].resample("ME").median()
    bs_m = merged["Bs"].resample("ME").median()
    ax1 = axes[1]
    ax1.plot(bz_m.index, bz_m.values, lw=1.3, color=PALETTE["Bz"], label="Bz moyen")
    ax1.fill_between(bs_m.index, 0, bs_m.values, alpha=0.35, color="red", label="Bs (composante sud)")
    ax1.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax1.set_ylabel("Bz / Bs (nT)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.35)
    
    # Epsilon de couplage
    eps_m = merged["epsilon_coupling"].resample("ME").median()
    ax2 = axes[2]
    ax2.fill_between(eps_m.index, eps_m, alpha=0.4, color=PALETTE["Vsw"])
    ax2.plot(eps_m.index, eps_m, lw=1.3, color=PALETTE["Vsw"])
    ax2.set_ylabel("ε couplage\n(km^(4/3)·s^(-4/3)·nT^(2/3))")
    ax2.set_title("Paramètre de couplage Newell ε = |V|^(4/3)·Bs^(2/3)", fontsize=9)
    ax2.set_xlabel("Date (UTC)")
    ax2.grid(alpha=0.35)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Fig8_VTEC_coupling.png",
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("    ✓ Fig8_VTEC_coupling.png")
else:
    print("    ⚠ Sautée (VTEC non disponible)")


# ───────────────────────────────────────────────────
# FIGURE 9 : Dashboard synthèse qualité des données
# ───────────────────────────────────────────────────
print("  → Figure 9 : Dashboard synthèse — qualité des données")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Rapport de qualité des données — OMNI 5 min (2015–2025)\n"
             "Pipeline de Data Engineering complet", fontsize=13, fontweight="bold")

gs9 = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# Panel A : Tableau des statistiques
ax_tab = fig.add_subplot(gs9[0, :])
ax_tab.axis("off")

table_data = []
for var in VARS:
    orig_total = len(df_raw)
    n_sentinel = sentinel_counts_raw[var]
    n_out = outlier_report[var]["n_outliers"]
    n_final_nan = df_interp[var].isna().sum()
    pct_valid = 100*(1 - n_final_nan/orig_total)
    stats_v = df_interp[var].dropna()
    table_data.append([
        var, UNITS[var],
        f"{orig_total:,}", f"{n_sentinel:,} ({100*n_sentinel/orig_total:.1f}%)",
        f"{n_out:,}", f"{n_final_nan:,}", f"{pct_valid:.1f}%",
        f"{stats_v.mean():.3f}", f"{stats_v.std():.3f}",
        f"{stats_v.min():.3f}", f"{stats_v.max():.3f}"
    ])

col_labels = ["Variable","Unité","Total brut","Sentinelles","Aberrants",
              "NaN résiduels","Couverture","Moyenne","Écart-type","Min","Max"]
tbl = ax_tab.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(AXES if r > 0 else "#1F2D5C")
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=TEXT)
ax_tab.set_title("A — Résumé statistique & qualité des données", fontweight="bold", fontsize=10, pad=8)

# Panel B : Taux de couverture par variable (barres)
ax_bar = fig.add_subplot(gs9[1, :2])
cov_vals = [100*(1 - df_interp[v].isna().mean()) for v in VARS]
bars = ax_bar.barh(VARS, cov_vals, color=[PALETTE[v] for v in VARS], alpha=0.8)
ax_bar.set_xlim(0, 100)
ax_bar.set_xlabel("Couverture après nettoyage (%)")
ax_bar.set_title("B — Taux de couverture finale", fontweight="bold")
for bar, val in zip(bars, cov_vals):
    ax_bar.text(val - 2, bar.get_y() + bar.get_height()/2,
               f"{val:.1f}%", va="center", ha="right", color="white", fontsize=9)
ax_bar.axvline(95, color="yellow", ls="--", lw=1.2, alpha=0.7, label="Seuil 95%")
ax_bar.legend(fontsize=8)
ax_bar.grid(axis="x", alpha=0.4)

# Panel C : Nombre d'aberrants par variable
ax_out = fig.add_subplot(gs9[1, 2:])
n_outs = [outlier_report[v]["n_outliers"] for v in VARS]
ax_out.bar(VARS, n_outs, color=[PALETTE[v] for v in VARS], alpha=0.8)
ax_out.set_ylabel("N aberrants (IQR×3)")
ax_out.set_title("C — Valeurs aberrantes détectées (IQR)", fontweight="bold")
for i, (v, n) in enumerate(zip(VARS, n_outs)):
    ax_out.text(i, n + 20, str(n), ha="center", fontsize=9, color=TEXT)
ax_out.grid(axis="y", alpha=0.4)

# Panel D : Timeline de la couverture VTEC (par mois)
ax_vtec = fig.add_subplot(gs9[2, :])
if not vtec_30.empty:
    vtec_cov = vtec_30["n_maps"].resample("ME").apply(
        lambda s: 100 * (s > 0).mean()
    )
    ax_vtec.fill_between(vtec_cov.index, vtec_cov.values, alpha=0.5, color=PALETTE["VTEC"])
    ax_vtec.plot(vtec_cov.index, vtec_cov.values, lw=1.5, color=PALETTE["VTEC"])
    ax_vtec.set_ylim(0, 105)
    ax_vtec.set_ylabel("Couverture VTEC (%)")
    ax_vtec.set_title("D — Couverture temporelle VTEC (GIM/IONEX) par mois", fontweight="bold")
    ax_vtec.set_xlabel("Date")
    ax_vtec.axhline(80, color="yellow", ls="--", lw=1, alpha=0.6, label="80%")
    ax_vtec.legend(fontsize=8)
else:
    ax_vtec.text(0.5, 0.5, "VTEC non disponible", ha="center", va="center",
                transform=ax_vtec.transAxes, fontsize=14, color="gray")
    ax_vtec.set_title("D — Couverture VTEC", fontweight="bold")
ax_vtec.grid(alpha=0.35)

plt.savefig(OUT_DIR / "Fig9_quality_dashboard.png",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("    ✓ Fig9_quality_dashboard.png")


# ═══════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PIPELINE COMPLET — RÉSUMÉ")
print("═"*65)
print(f"\n  Fichiers CSV produits :")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"    • {f.name}  ({f.stat().st_size/1024:.0f} Ko)")
print(f"\n  Figures EDA produites :")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"    • {f.name}  ({f.stat().st_size/1024:.0f} Ko)")

print(f"""
  OMNI 5 min  : {len(df_raw):,} mesures brutes
  OMNI 30 min : {len(omni_30):,} bins nettoyés
  VTEC 30 min : {len(vtec_30):,} bins GIM (Beni Mellal)
  Tempêtes Bz : {len(df_storms)} événements (Bz < {BZ_THRESH} nT)
  ε couplage  : calculé (Newell simplifié)
""")
print("  ✅ Pipeline terminé avec succès.\n")
