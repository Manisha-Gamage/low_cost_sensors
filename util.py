#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# make folder if not there
def make_out_folder(out):
    os.makedirs(out, exist_ok=True)


# rmse
def get_rmse(x, y):
    return float(np.sqrt(mean_squared_error(x, y)))


# collect metrics in one place
def get_metrics(x, y):
    return {
        "RMSE": get_rmse(x, y),
        "MAE": float(mean_absolute_error(x, y)),
        "R2": float(r2_score(x, y)),
        "N": int(len(x)),
    }


# print metrics
def show_metrics(x, y):
    print(f"\n{x}")
    for a, b in y.items():
        if isinstance(b, float):
            print(f"  {a}: {b:.3f}")
        else:
            print(f"  {a}: {b}")


# convert cols to numeric but not datetime
def fix_num(x):
    y = x.copy()
    for a in y.columns:
        if a != "datetime":
            y[a] = pd.to_numeric(y[a], errors="coerce")
    return y


# load naps
def load_naps(input):
    x = pd.read_excel(input, sheet_name="NAPS DATA")
    # removing unit row
    x = x.iloc[1:].copy()
    x = x.rename(columns={"Column1": "datetime"})
    # datetime fix
    x["datetime"] = pd.to_datetime(x["datetime"], dayfirst=True, errors="coerce")

    # keep only needed cols
    y = ["datetime","SO2 ppb", "NO ppb", "NO2 ppb", "NOX ppb", "CO ppm", "O3 ppb",
        "PM25 Ug/m3", "Temp", "RH" ]
    x = x[y]
    x = fix_num(x)
    x = x.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return x


# load compact
def load_compact(input, x=6):
    y = pd.read_excel(input, sheet_name="Compact Station")
    # removing unit row
    y = y.iloc[1:].copy()
    y = y.rename(columns={"Datetime": "datetime"})
    y["datetime"] = pd.to_datetime(y["datetime"], errors="coerce")

    if getattr(y["datetime"].dt, "tz", None) is not None:
        y["datetime"] = y["datetime"].dt.tz_localize(None)

    # keep only needed cols
    x_cols = ["datetime","SO2-1", "CO-200", "O3-5", "NO-1", "NO2-2","T", "RH", "PM-2.5"]
    y = y[x_cols]
    y = fix_num(y)
    y = y.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # change conversion
    y["NO_ppb"] = y["NO-1"] * 1000.0
    y["NO2_ppb"] = y["NO2-2"] * 1000.0
    y["O3_ppb"] = y["O3-5"] * 1000.0
    y["SO2_ppb"] = y["SO2-1"] * 1000.0

    # co already in ppm
    y["CO_ppm"] = y["CO-200"]

    # warmup hour
    y["warmup_minutes"] = (y["datetime"] - y["datetime"].min()).dt.total_seconds() / 60.0
    y["warmup_flag_6h"] = (y["warmup_minutes"] < x * 60).astype(int)

    return y

# compact to 15 min
def compact_to_15(x):
    # x - compact raw
    y = x.set_index("datetime").sort_index()

    a = [
        "SO2_ppb", "CO_ppm", "O3_ppb", "NO_ppb", "NO2_ppb",
        "T", "RH", "PM-2.5", "warmup_flag_6h"
    ]

    y = y[a].resample("15min").mean().reset_index()
    y = y.rename(columns={"T": "T_compact", "RH": "RH_compact"})

    return y


# merge naps and compact
def merge_data(x, y):
    # x is naps
    # y is compact 15 min
    out = pd.merge(x, y, on="datetime", how="inner")
    out = out.rename(columns={"Temp": "Temp_ref", "RH": "RH_ref"})
    out = out.sort_values("datetime").reset_index(drop=True)

    return out

# plot line
def save_line(x, y, out, a, b):
    plt.figure(figsize=(12, 5))
    for c in y:
        plt.plot(x["datetime"], x[c], label=c)

    plt.title(a)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, b), dpi=200)
    plt.show()
    plt.close()

# plot scatter
def save_scatter(x, y, a, out, b, c):
    plt.figure(figsize=(6, 5))
    plt.scatter(x[y], x[a], s=12, alpha=0.6)
    plt.title(b)
    plt.xlabel(y)
    plt.ylabel(a)
    plt.tight_layout()
    plt.savefig(os.path.join(out, c), dpi=200)
    plt.show()
    plt.close()

# Q1
@dataclass
class Q1Result:
    x: dict
    y: dict
    a: dict | None
    b: dict | None
    c: dict | None
    d: dict | None
    out: pd.DataFrame
    m1: LinearRegression
    m2: LinearRegression


def run_q1(x, y=True, a=False):
    # x - merged data
    out = x[["datetime", "PM25 Ug/m3", "PM-2.5", "RH_ref", "Temp_ref", "warmup_flag_6h"]].dropna().copy()
    out = out.sort_values("datetime")

    if a:
        out = out[out["warmup_flag_6h"] == 0].copy()

    z = ["PM-2.5", "RH_ref"]
    if y:
        z.append("Temp_ref")

    s = int(len(out) * 0.70)
    x1 = out.iloc[:s].copy()
    y1 = out.iloc[s:].copy()

    m1 = LinearRegression()
    m1.fit(x1[["PM-2.5"]], x1["PM25 Ug/m3"])

    m2 = LinearRegression()
    m2.fit(x1[z], x1["PM25 Ug/m3"])

    y1["pred_raw"] = m1.predict(y1[["PM-2.5"]])
    y1["pred_corr"] = m2.predict(y1[z])

    p1 = get_metrics(y1["PM25 Ug/m3"], y1["pred_raw"])
    p2 = get_metrics(y1["PM25 Ug/m3"], y1["pred_corr"])

    r1 = y1[y1["RH_ref"] > 75].copy()
    r2 = y1[y1["RH_ref"] > 85].copy()

    q1 = get_metrics(r1["PM25 Ug/m3"], r1["pred_raw"]) if len(r1) > 3 else None
    q2 = get_metrics(r1["PM25 Ug/m3"], r1["pred_corr"]) if len(r1) > 3 else None
    q3 = get_metrics(r2["PM25 Ug/m3"], r2["pred_raw"]) if len(r2) > 3 else None
    q4 = get_metrics(r2["PM25 Ug/m3"], r2["pred_corr"]) if len(r2) > 3 else None

    return Q1Result(
        x=p1,
        y=p2,
        a=q1,
        b=q2,
        c=q3,
        d=q4,
        out=y1,
        m1=m1,
        m2=m2,
    )


def q1_rh_bias(x):
    y = x[["datetime", "PM25 Ug/m3", "PM-2.5", "RH_ref"]].dropna().copy()
    y["bias_compact_minus_ref"] = y["PM-2.5"] - y["PM25 Ug/m3"]

    a = [0, 60, 75, 85, 100]
    b = ["<60%", "60-75%", "75-85%", "85-100%"]

    y["RH_band"] = pd.cut(y["RH_ref"], bins=a, labels=b, include_lowest=True)

    out = (
        y.groupby("RH_band", observed=False)["bias_compact_minus_ref"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    return out


# Q2
@dataclass
class Q2Result:
    x: dict
    y: dict
    a: int
    b: int
    out: pd.DataFrame
    z: pd.DataFrame


def compare_q2(x, y, a=15, b=0):
    # x - compact raw
    # y - naps
    # a - moving average window
    # b is lag in min
    out = x[["datetime", "NO2_ppb"]].dropna().copy().sort_values("datetime")
    out = out.set_index("datetime").resample("1min").mean().interpolate(limit=3)

    out["NO2_smoothed"] = out["NO2_ppb"].rolling(a, min_periods=1).mean()
    out = out.reset_index()
    out["datetime"] = out["datetime"] + pd.to_timedelta(b, unit="min")

    z = out.set_index("datetime")[["NO2_smoothed"]].resample("15min").mean().reset_index()

    p = y[["datetime", "NO2 ppb"]].dropna().copy()
    out = pd.merge(p, z, on="datetime", how="inner").dropna()

    return out


def run_q2(x, y):
    # x - compact raw
    # y - naps
    z = (
        x.set_index("datetime")[["NO2_ppb"]]
        .resample("15min").mean()
        .reset_index()
    )

    out = pd.merge(
        y[["datetime", "NO2 ppb"]].dropna(),
        z,
        on="datetime",
        how="inner"
    ).dropna()

    p1 = {
        "correlation": float(out["NO2 ppb"].corr(out["NO2_ppb"])),
        "RMSE": get_rmse(out["NO2 ppb"], out["NO2_ppb"]),
        "MAE": float(mean_absolute_error(out["NO2 ppb"], out["NO2_ppb"])),
        "N": int(len(out)),
    }

    all_rows = []
    best_score = None
    best_w = None
    best_l = None
    best_out = None

    for a in [1, 5, 10, 15, 20, 30, 45, 60]:
        for b in range(-30, 31, 1):
            c = compare_q2(x, y, a=a, b=b)

            if len(c) < 20:
                continue

            d = float(c["NO2 ppb"].corr(c["NO2_smoothed"]))
            e = get_rmse(c["NO2 ppb"], c["NO2_smoothed"])
            f = float(mean_absolute_error(c["NO2 ppb"], c["NO2_smoothed"]))

            all_rows.append({
                "window_min": a,
                "lag_min": b,
                "correlation": d,
                "RMSE": e,
                "MAE": f,
                "N": len(c),
            })

            score = (d, -e)
            if best_score is None or score > best_score:
                best_score = score
                best_w = a
                best_l = b
                best_out = c.copy()

    g = pd.DataFrame(all_rows).sort_values(["correlation", "RMSE"], ascending=[False, True])

    p2 = {
        "correlation": float(best_out["NO2 ppb"].corr(best_out["NO2_smoothed"])),
        "RMSE": get_rmse(best_out["NO2 ppb"], best_out["NO2_smoothed"]),
        "MAE": float(mean_absolute_error(best_out["NO2 ppb"], best_out["NO2_smoothed"])),
        "N": int(len(best_out)),
    }

    return Q2Result(
        x=p1,
        y=p2,
        a=int(best_w),
        b=int(best_l),
        out=best_out,
        z=g,
    )


# Q3
@dataclass
class Q3Result:
    x: dict
    y: dict
    a: dict
    b: dict | None
    c: dict | None
    d: dict | None
    e: dict
    out: pd.DataFrame


def run_q3(x, y=False):
    out = x[
        [
            "datetime",
            "NO2 ppb",
            "NO2_ppb",
            "O3_ppb",
            "Temp_ref",
            "RH_ref",
            "warmup_flag_6h",
            "O3 ppb",
        ]
    ].dropna().copy().sort_values("datetime")

    if y:
        out = out[out["warmup_flag_6h"] == 0].copy()

    out["hour"] = out["datetime"].dt.hour
    out["afternoon"] = out["hour"].between(12, 17).astype(int)
    out["O3_afternoon_interaction"] = out["O3_ppb"] * out["afternoon"]

    s = int(len(out) * 0.70)
    x1 = out.iloc[:s].copy()
    y1 = out.iloc[s:].copy()

    z1 = x1["NO2 ppb"]
    z2 = y1["NO2 ppb"]

    m1 = LinearRegression().fit(x1[["NO2_ppb"]], z1)
    y1["pred_m1"] = m1.predict(y1[["NO2_ppb"]])

    a1 = ["NO2_ppb", "O3_ppb", "Temp_ref", "RH_ref"]
    m2 = LinearRegression().fit(x1[a1], z1)
    y1["pred_m2"] = m2.predict(y1[a1])

    a2 = ["NO2_ppb", "O3_ppb", "O3_afternoon_interaction", "Temp_ref", "RH_ref"]
    m3 = LinearRegression().fit(x1[a2], z1)
    y1["pred_m3"] = m3.predict(y1[a2])

    p1 = get_metrics(z2, y1["pred_m1"])
    p2 = get_metrics(z2, y1["pred_m2"])
    p3 = get_metrics(z2, y1["pred_m3"])

    a = y1[y1["afternoon"] == 1].copy()

    q1 = get_metrics(a["NO2 ppb"], a["pred_m1"]) if len(a) > 3 else None
    q2 = get_metrics(a["NO2 ppb"], a["pred_m2"]) if len(a) > 3 else None
    q3 = get_metrics(a["NO2 ppb"], a["pred_m3"]) if len(a) > 3 else None

    c1 = float(out["NO2_ppb"].corr(out["O3_ppb"]))
    c2 = float(out["NO2 ppb"].corr(out["O3 ppb"]))

    b1 = out[out["afternoon"] == 1].copy()
    c3 = float(b1["NO2_ppb"].corr(b1["O3_ppb"])) if len(b1) > 3 else np.nan
    c4 = float(b1["NO2 ppb"].corr(b1["O3 ppb"])) if len(b1) > 3 else np.nan

    c_all = {
        "compact_NO2_vs_compact_O3_alltime": c1,
        "reference_NO2_vs_reference_O3_alltime": c2,
        "compact_NO2_vs_compact_O3_afternoon": c3,
        "reference_NO2_vs_reference_O3_afternoon": c4,
    }

    return Q3Result(
        x=p1,
        y=p2,
        a=p3,
        b=q1,
        c=q2,
        d=q3,
        e=c_all,
        out=y1,
    )

