# -*- coding: utf-8 -*-
"""Fichier daily : statistiques journalieres VTEC + proxies de meteo de l'espace.

Aligne la partie << proxies SW >> du cahier des charges en produisant un fichier
quotidien combinant :

  - les statistiques journalieres du VTEC au-dessus de Beni Mellal
    (agregees depuis output/VTEC_GIM_30min_BeniMellal.csv) ;
  - les indices solaires/geomagnetiques canoniques F10.7, Kp et Dst,
    telecharges depuis leurs centres de donnees de reference.

Sources des indices (donnees publiques) :
  - F10.7 (flux radio solaire 10.7 cm, s.f.u.) et Kp / Ap : GFZ Helmholtz Centre
    for Geosciences, fichier << Kp_ap_Ap_SN_F107_since_1932.txt >>.
    Matzka et al. (2021), https://doi.org/10.5880/Kp.0001
  - Dst (indice equatorial, nT) : WDC for Geomagnetism, Kyoto
    (tables mensuelles final / provisional / realtime).

Sortie : output/DAILY_VTEC_SW_indices.csv (une ligne par jour UT).

Reproduction :
    python 02_daily_sw_indices_PFA.py

Les telechargements sont mis en cache dans data_cache/ : un second lancement est
hors-ligne. Les jours sans donnee restent vides (NaN), aucune valeur n'est inventee.
"""
from __future__ import annotations

import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
CACHE = ROOT / "data_cache"
CACHE.mkdir(exist_ok=True)

VTEC_30MIN = OUTPUT / "VTEC_GIM_30min_BeniMellal.csv"
OUT_CSV = OUTPUT / "DAILY_VTEC_SW_indices.csv"

GFZ_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"

# Les centres de donnees scientifiques presentent parfois une chaine de
# certificats incomplete sur ce poste ; on desactive la verification TLS
# uniquement pour ces telechargements de donnees publiques en lecture seule.
_SSL = ssl._create_unverified_context()
_HEADERS = {"User-Agent": "Mozilla/5.0 (PFA-VTEC daily indices)"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download(url: str, dest: Path) -> Path:
    """Telecharge url vers dest si absent du cache. Renvoie le chemin local."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _log(f"  download {url}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=90, context=_SSL) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


# ---------------------------------------------------------------------------
# 1. Statistiques journalieres du VTEC
# ---------------------------------------------------------------------------
def daily_vtec() -> pd.DataFrame:
    df = pd.read_csv(VTEC_30MIN)
    tcol = df.columns[0]
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    s = pd.Series(df["vtec_mean"].to_numpy(), index=t).dropna()
    g = s.groupby(s.index.normalize())
    out = pd.DataFrame({
        "vtec_n": g.size(),
        "vtec_mean": g.mean(),
        "vtec_median": g.median(),
        "vtec_std": g.std(),
        "vtec_min": g.min(),
        "vtec_max": g.max(),
    })
    out.index = out.index.tz_localize(None).normalize()
    out.index.name = "date"
    _log(f"VTEC daily : {len(out)} jours ({out.index.min().date()} -> {out.index.max().date()})")
    return out


# ---------------------------------------------------------------------------
# 2. F10.7, Kp, Ap  (GFZ)
# ---------------------------------------------------------------------------
def daily_gfz() -> pd.DataFrame:
    path = _download(GFZ_URL, CACHE / "Kp_ap_Ap_SN_F107_since_1932.txt")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 28:
            continue
        y, m, d = int(p[0]), int(p[1]), int(p[2])
        kp = [float(x) for x in p[7:15]]          # Kp1..Kp8
        ap_daily = float(p[23])                    # Ap
        f107obs = float(p[25])                     # F10.7obs
        kp = [np.nan if v < 0 else v for v in kp]
        rows.append((pd.Timestamp(y, m, d), np.nanmean(kp) if any(~np.isnan(kp)) else np.nan,
                     np.nanmax(kp) if any(~np.isnan(kp)) else np.nan,
                     np.nan if ap_daily < 0 else ap_daily,
                     np.nan if f107obs < 0 else f107obs))
    df = pd.DataFrame(rows, columns=["date", "kp_mean", "kp_max", "ap", "f107_obs"]).set_index("date")
    df["kp_mean"] = df["kp_mean"].round(3)
    _log(f"GFZ (F10.7/Kp/Ap) : {len(df)} jours")
    return df


# ---------------------------------------------------------------------------
# 3. Dst  (WDC Kyoto, tables mensuelles)
# ---------------------------------------------------------------------------
def _parse_kyoto_month(text: str) -> dict[int, list[float]]:
    """Renvoie {jour: [24 valeurs horaires Dst]} depuis le bloc <pre>."""
    m = re.search(r"<pre.*?>(.*?)</pre>", text, re.S | re.I)
    body = m.group(1) if m else text
    body = re.sub(r"<[^>]+>", "", body)
    days: dict[int, list[float]] = {}
    for line in body.splitlines():
        ints = re.findall(r"-?\d+", line)
        # une ligne de donnees = jour + 24 valeurs horaires
        if len(ints) >= 25:
            day = int(ints[0])
            vals = [float(v) for v in ints[1:25]]
            vals = [np.nan if abs(v) >= 9000 else v for v in vals]
            if 1 <= day <= 31:
                days[day] = vals
    return days


def _fetch_kyoto_month(year: int, month: int) -> dict[int, list[float]]:
    ym = f"{year:04d}{month:02d}"
    cache = CACHE / f"dst_{ym}.txt"
    if cache.exists() and cache.stat().st_size > 0:
        return _parse_kyoto_month(cache.read_text(encoding="utf-8", errors="ignore"))
    for kind in ("dst_final", "dst_provisional", "dst_realtime"):
        url = f"https://wdc.kugi.kyoto-u.ac.jp/{kind}/{ym}/index.html"
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            raw = urllib.request.urlopen(req, timeout=60, context=_SSL).read()
        except Exception:
            continue
        text = raw.decode("shift_jis", errors="ignore")
        parsed = _parse_kyoto_month(text)
        if parsed:
            cache.write_text(text, encoding="utf-8")
            _log(f"  Dst {ym} <- {kind}")
            return parsed
    _log(f"  Dst {ym} : indisponible")
    return {}


def daily_dst(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for ts in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        days = _fetch_kyoto_month(ts.year, ts.month)
        for day, vals in days.items():
            try:
                date = pd.Timestamp(ts.year, ts.month, day)
            except ValueError:
                continue
            arr = np.array(vals, dtype=float)
            if np.isnan(arr).all():
                continue
            rows.append((date, float(np.nanmean(arr)), float(np.nanmin(arr))))
    df = pd.DataFrame(rows, columns=["date", "dst_mean", "dst_min"]).set_index("date")
    df["dst_mean"] = df["dst_mean"].round(1)
    _log(f"Dst (Kyoto) : {len(df)} jours")
    return df


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------
def main() -> int:
    if not VTEC_30MIN.exists():
        _log(f"ERREUR : {VTEC_30MIN} introuvable. Lancez d'abord 01_data_engineering_PFA.py.")
        return 1

    vtec = daily_vtec()
    start, end = vtec.index.min(), vtec.index.max()

    gfz = daily_gfz()
    dst = daily_dst(start, end)

    # Grille journaliere continue sur la periode VTEC exploitable
    idx = pd.date_range(start, end, freq="D")
    out = pd.DataFrame(index=idx)
    out.index.name = "date"
    out = out.join(vtec).join(gfz).join(dst)

    cols = ["vtec_n", "vtec_mean", "vtec_median", "vtec_std", "vtec_min", "vtec_max",
            "f107_obs", "kp_mean", "kp_max", "ap", "dst_mean", "dst_min"]
    out = out[cols]
    out["vtec_n"] = out["vtec_n"].fillna(0).astype(int)
    for c in ["vtec_mean", "vtec_median", "vtec_std", "vtec_min", "vtec_max"]:
        out[c] = out[c].round(3)

    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=True, date_format="%Y-%m-%d")

    cov = out[["f107_obs", "kp_mean", "dst_mean"]].notna().mean().mul(100).round(1)
    _log("")
    _log(f"Ecrit : {OUT_CSV}  ({len(out)} jours, {start.date()} -> {end.date()})")
    _log(f"Couverture indices : F10.7 {cov['f107_obs']}%  Kp {cov['kp_mean']}%  Dst {cov['dst_mean']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
