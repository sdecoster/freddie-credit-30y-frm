"""One-shot script to add fixed overlay/bar-chart cells + a metrics-dump cell
to notebooks/partA_survival.ipynb, before the final 'Saved artifacts' markdown.

Idempotent: removes any previously-inserted cells with matching marker ids
before inserting fresh ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "partA_survival.ipynb"

# Marker prefix so re-running this script replaces, not duplicates.
MARKER = "partA_presentation_addendum"


def cell_code(src: str, cid: str) -> dict:
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def cell_md(src: str, cid: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


OVERLAY_MD = """### Per-vintage hazard overlay — the calendar-time smoking gun

If the seasoning hump in the aggregate hazard reflects a *calendar* effect (rate
cycle / refi window) rather than an intrinsic loan-age effect, then each
vintage's hump should appear at a different loan age — specifically the age at
which that vintage was exposed to the rate trough.

Recent vintages (2022) are dropped because their observable history is shorter
than the analysis horizon, so the right-tail estimator is unstable.
"""

OVERLAY_CODE = """# Per-vintage hazard overlay. Calendar smoking gun.
import gc, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm
from lifelines.utils import survival_table_from_events

SMOOTH = 3
VINTAGE_MIN_MAX_MONTHS = 48  # vintage must be observable for >= 48 mo

vintage_counts = (
    loans
    .filter(pl.col("event_time_months").is_not_null())
    .group_by("vintage_year")
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") >= 20_000)
    .sort("vintage_year")
)
vintages = sorted(vintage_counts["vintage_year"].to_list())

cmap = cm.get_cmap("plasma", len(vintages))
colors = {v: cmap(i) for i, v in enumerate(vintages)}

fig, ax = plt.subplots(figsize=(12, 6))
plotted_vintages = []

for vint in vintages:
    sub = (
        loans
        .filter((pl.col("vintage_year") == vint) & pl.col("event_time_months").is_not_null())
        .select(["event_time_months", "prepay_observed"])
        .to_pandas()
    )
    T = sub["event_time_months"]; E = sub["prepay_observed"]
    tbl = survival_table_from_events(T, E)

    # Exclude vintages whose data ends before the analysis horizon.
    if int(tbl.index.max()) < VINTAGE_MIN_MAX_MONTHS:
        del sub, T, E, tbl; gc.collect()
        continue

    hz = (tbl["observed"] / tbl["at_risk"].replace(0, np.nan)) \\
            .rolling(SMOOTH, center=True, min_periods=1).mean()

    initial_at_risk = int(tbl["at_risk"].iloc[0])
    vintage_floor = max(AT_RISK_FLOOR, initial_at_risk // 10)
    cutoff_idx = tbl.index[tbl["at_risk"] < vintage_floor]
    max_t = int(cutoff_idx[0]) if len(cutoff_idx) else EFFECTIVE_MAX_MONTHS
    max_t = min(max_t, EFFECTIVE_MAX_MONTHS)

    ax.plot(hz.loc[:max_t].index, hz.loc[:max_t].values * 100,
            color=colors[vint], lw=1.5, label=str(vint), alpha=0.85)
    plotted_vintages.append(vint)
    del sub, T, E, tbl, hz; gc.collect()

ax.set_xlabel("Months from origination")
ax.set_ylabel(f"Prepayment hazard rate % ({SMOOTH}-month rolling avg)")
ax.set_title("Prepayment hazard by vintage — aligned by loan age\\n"
             "Calendar-time driver: hump shifts LEFT as vintage advances")
ax.set_xlim(0, EFFECTIVE_MAX_MONTHS); ax.set_ylim(0)
ax.legend(title="Vintage", ncol=3, fontsize=7, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "partA_hazard_vintage_overlay.png", dpi=150)
plt.show(); plt.close(fig); gc.collect()

print(f"plotted {len(plotted_vintages)} vintages: {plotted_vintages}")
"""

BAR_MD = """### Hazard at a fixed loan age, by vintage

For each vintage, the average prepayment hazard at loan ages 21-27 months
(centred on month 24). Each bar is labelled with the calendar year that vintage
was 24 months old. If the calendar story is right, the tall bars should
correspond to refi-incentive calendar years (2003, 2012-13, 2020-21).
"""

BAR_CODE = """# Hazard at a fixed loan age, by vintage. Quantitative smoking gun.
import gc, numpy as np, pandas as pd, matplotlib.pyplot as plt
from lifelines.utils import survival_table_from_events

TARGET_AGE = 24
BAND       = 6

vintages = sorted(
    loans
    .filter(pl.col("event_time_months").is_not_null())
    .group_by("vintage_year")
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") >= 5_000)
    ["vintage_year"].to_list()
)

rows = []
for vint in vintages:
    sub = (
        loans
        .filter((pl.col("vintage_year") == vint) & pl.col("event_time_months").is_not_null())
        .select(["event_time_months", "prepay_observed"])
        .to_pandas()
    )
    T = sub["event_time_months"]; E = sub["prepay_observed"]
    tbl = survival_table_from_events(T, E)
    lo, hi = TARGET_AGE - BAND // 2, TARGET_AGE + BAND // 2
    band = tbl.loc[(tbl.index >= lo) & (tbl.index <= hi)]

    if len(band) == 0 or band["at_risk"].min() < AT_RISK_FLOOR:
        del sub, T, E, tbl, band; gc.collect(); continue

    hz = (band["observed"] / band["at_risk"].replace(0, np.nan)).mean()
    rows.append({"vintage_year": vint,
                 "calendar_year": vint + TARGET_AGE // 12,
                 "hazard_pct": hz * 100})
    del sub, T, E, tbl, band; gc.collect()

df_hz = pd.DataFrame(rows).sort_values("vintage_year")
df_hz.to_csv(FIGURES_DIR / "partA_hazard_fixed_age_by_vintage.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_hz["vintage_year"], df_hz["hazard_pct"],
       color="steelblue", width=0.7, edgecolor="white")
for _, row in df_hz.iterrows():
    ax.text(row["vintage_year"], row["hazard_pct"] + 0.005,
            f"cal {int(row['calendar_year'])}",
            ha="center", va="bottom", fontsize=7, color="dimgray")

ax.set_xlabel("Vintage year")
ax.set_ylabel(f"Prepayment hazard at month ~{TARGET_AGE} (%)")
ax.set_title(f"Prepayment hazard at loan age ~{TARGET_AGE} months, by vintage\\n"
             f"Labels = calendar year — tall bars should align with refi windows")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "partA_hazard_fixed_age_by_vintage.png", dpi=150)
plt.show(); plt.close(fig)
print(df_hz.to_string(index=False))
del df_hz; gc.collect()
"""

DISCRETE_HAZARD_MD = """### Aggregate discrete monthly hazard (single panel)

Single-panel discrete monthly hazard plot for the presentation. The smoothed
companion plot in cell 7 (`partA_hazard_aggregate.png`) is left intact for the
notebook reader, but the deck uses this one.
"""

DISCRETE_HAZARD_CODE = """# Discrete-only aggregate hazard, single panel, for the deck.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
disc = discrete_hazard[:EFFECTIVE_MAX_MONTHS]
ax.plot(disc.index, disc.values * 100, lw=1.0, color="tab:blue")
ax.set_xlabel("Months from origination")
ax.set_ylabel("Monthly prepay hazard (%)")
ax.set_title(f"Aggregate prepayment hazard, vintages {YEARS[0]}-{YEARS[-1]}")
ax.set_xlim(0, EFFECTIVE_MAX_MONTHS); ax.set_ylim(0)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "partA_hazard_discrete.png", dpi=150)
plt.show(); plt.close(fig)
"""


KM_PER_COVARIATE_MD = """### KM-only per-covariate panels (for the deck)

Single-panel KM-only renders for each of the seven covariates. The original
`plot_stratified()` (cells 11-23) saves a two-panel KM+hazard figure; the deck
uses the cleaner KM-only versions saved here.

Filename pattern: `figures/partA_strat_{stem}_km.png`.
"""

KM_PER_COVARIATE_CODE = """# KM-only per-covariate panels. Memory-conscious lifetable approach.
import gc, numpy as np, matplotlib.pyplot as plt
from lifelines.utils import survival_table_from_events

def _km_lifetable(d, e, max_m):
    t = survival_table_from_events(d, e)
    t = t[t.index <= max_m]
    h = (t["observed"] / t["at_risk"]).rename("hazard")
    s = (1.0 - h).cumprod().rename("survival")
    return s

def plot_km_only(loans_df, group_col, save_stem, bin_cfg=None, explicit_groups=None,
                 title=None, sample_per_group=SAMPLE_PER_GROUP):
    if bin_cfg is not None:
        df = (loans_df.filter(pl.col(group_col).is_not_null())
              .with_columns(pl.col(group_col)
                              .cut(breaks=bin_cfg["breaks"], labels=bin_cfg["labels"])
                              .alias("__grp")))
        ordered_groups = bin_cfg["labels"]
    elif explicit_groups is not None:
        cols = []
        ordered_groups = list(explicit_groups.keys())
        for label, mask in explicit_groups.items():
            cols.append(pl.when(mask).then(pl.lit(label)).otherwise(None))
        df = (loans_df.with_columns(pl.coalesce(cols).alias("__grp"))
              .filter(pl.col("__grp").is_not_null()))
    else:
        raise ValueError("provide bin_cfg or explicit_groups")

    if sample_per_group is not None:
        parts = []
        for grp in ordered_groups:
            sub = df.filter(pl.col("__grp") == grp)
            if sub.height > sample_per_group:
                sub = sub.sample(n=sample_per_group, seed=0)
            parts.append(sub)
        df = pl.concat(parts)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(ordered_groups)))
    for color, label in zip(cmap, ordered_groups):
        sub = df.filter(pl.col("__grp") == label)
        if sub.height < 100:
            continue
        d = sub["event_time_months"].to_numpy()
        e = sub["prepay_observed"].to_numpy()
        s = _km_lifetable(d, e, EFFECTIVE_MAX_MONTHS)
        ax.step(s.index, s.values, where="post", color=color, lw=1.6,
                label=f"{label} (n={sub.height:,})")
        del sub, d, e, s; gc.collect()

    ax.set_xlim(0, EFFECTIVE_MAX_MONTHS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months from origination")
    ax.set_ylabel("P(not prepaid)")
    ax.set_title(title or f"Prepayment survival by {group_col}")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"partA_strat_{save_stem}_km.png", dpi=150)
    plt.show(); plt.close(fig)
    del df; gc.collect()

# 7 covariates, KM-only.
plot_km_only(loans, "fico",        "fico",        bin_cfg=COVARIATE_BINS["fico"],
             title="KM survival by FICO")
plot_km_only(loans, "ltv",         "ltv",         bin_cfg=COVARIATE_BINS["ltv"],
             title="KM survival by LTV")
plot_km_only(loans, "dti",         "dti",         bin_cfg=COVARIATE_BINS["dti"],
             title="KM survival by DTI")
plot_km_only(loans, "orig_rate",   "orig_rate",   bin_cfg=COVARIATE_BINS["orig_rate"],
             title="KM survival by original interest rate")
plot_km_only(loans, "vintage_year","vintage",
             explicit_groups={lbl: pl.col("vintage_year").is_in(yrs)
                              for lbl, yrs in VINTAGE_GROUPS.items()},
             title="KM survival by vintage group")
plot_km_only(loans, "loan_purpose","loan_purpose",
             explicit_groups=LOAN_PURPOSE_GROUPS,
             title="KM survival by loan purpose")
plot_km_only(loans, "channel",     "channel",
             explicit_groups=CHANNEL_GROUPS,
             title="KM survival by origination channel")
print("KM-only stratified panels saved.")
"""


COX_BINFREE_MD = """### Bin-free Cox: strength of stratification across all 7 covariates

Replaces the binned-log-rank bar chart for presentation use. For each covariate
we fit a univariate Cox PH model and read off the likelihood-ratio chi² of the
fitted model against the null. Numerics enter continuously; categoricals enter
as factor (one-hot) terms, so the chi² is "bin-free" in the sense that no
arbitrary numeric thresholds were chosen.

Caveat (noted on the slide): the chi² of a factor Cox depends on the number of
levels — a 17-level vintage factor gets more degrees of freedom than a 3-level
loan_purpose factor. The ranking is robust at the top end but mid-rank
distinctions should be read as "comparable, not identical."

Saves a CSV (`partA_stratification_strength_cox.csv`) and a bar chart
(`partA_stratification_strength_cox.png`).
"""

COX_BINFREE_CODE = """# Bin-free Cox: univariate LR chi-squared per covariate.
import gc, pandas as pd, matplotlib.pyplot as plt
from lifelines import CoxPHFitter

NUMERIC_COVARS = ["fico", "ltv", "dti", "orig_rate"]
CATEGORICAL_COVARS = ["vintage_year", "loan_purpose", "channel"]

cox_rows = []

for col in NUMERIC_COVARS:
    sub = (loans.select(["event_time_months", "prepay_observed", col])
           .drop_nulls().to_pandas())
    if len(sub) > 500_000:
        sub = sub.sample(500_000, random_state=0)
    cph = CoxPHFitter().fit(sub, "event_time_months", "prepay_observed", formula=col)
    lrt = cph.log_likelihood_ratio_test()
    cox_rows.append({
        "covariate":   col,
        "kind":        "numeric (continuous)",
        "n_terms":     1,
        "n":           len(sub),
        "cox_lr_chi2": round(float(lrt.test_statistic), 1),
        "p_value":     float(lrt.p_value),
    })
    del sub, cph, lrt; gc.collect()

for col in CATEGORICAL_COVARS:
    sub = (loans.select(["event_time_months", "prepay_observed", col])
           .drop_nulls().to_pandas())
    if len(sub) > 500_000:
        sub = sub.sample(500_000, random_state=0)
    # Treat as categorical factor.
    sub[col] = sub[col].astype("category")
    n_levels = sub[col].nunique()
    formula = f"C({col})"
    cph = CoxPHFitter().fit(sub, "event_time_months", "prepay_observed", formula=formula)
    lrt = cph.log_likelihood_ratio_test()
    cox_rows.append({
        "covariate":   col.replace("_year", ""),
        "kind":        f"categorical ({n_levels} levels)",
        "n_terms":     int(n_levels - 1),
        "n":           len(sub),
        "cox_lr_chi2": round(float(lrt.test_statistic), 1),
        "p_value":     float(lrt.p_value),
    })
    del sub, cph, lrt; gc.collect()

cox_strength = (pd.DataFrame(cox_rows)
                  .sort_values("cox_lr_chi2", ascending=False)
                  .reset_index(drop=True))
cox_strength.to_csv(FIGURES_DIR / "partA_stratification_strength_cox.csv", index=False)
print(cox_strength.to_string(index=False))

# Bar chart.
fig, ax = plt.subplots(figsize=(9, 4))
order = cox_strength.sort_values("cox_lr_chi2")
colors = ["tab:blue" if "numeric" in k else "tab:orange" for k in order["kind"]]
ax.barh(order["covariate"], order["cox_lr_chi2"], color=colors, alpha=0.85)
for i, (cov, val) in enumerate(zip(order["covariate"], order["cox_lr_chi2"])):
    ax.text(val, i, f"  {val:,.0f}", va="center", fontsize=9)
ax.set_xlabel("Cox univariate LR chi-squared")
ax.set_title("Strength of stratification (bin-free): Cox LR chi-squared per covariate\\n"
             "Blue = numeric (continuous); orange = categorical (factor)")
ax.grid(axis="x", alpha=0.3)
# Add a legend / footnote about factor df.
ax.text(0.98, 0.02,
        "Factor chi-sq grows with #levels;\\nmid-rank ordering is comparable not identical.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="dimgray",
        style="italic")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "partA_stratification_strength_cox.png", dpi=150)
plt.show(); plt.close(fig)
gc.collect()
"""


METRICS_MD = """### Metrics dump

Writes key numbers from the analysis to `figures/partA_metrics.json` so the
presentation can quote accurate values. Must be run **after** all cells above
(KM, hazard, stratified, AJ).
"""

METRICS_CODE = """# Dump key numerics to figures/partA_metrics.json for the presentation deck.
import json
import numpy as np

metrics = {}

# --- Loan counts --------------------------------------------------------------
metrics["loans_in_analysis"] = int(loans.height)
metrics["years_in_sample"]   = [int(YEARS[0]), int(YEARS[-1])]

from src.credit_data import load_origination
orig_count = int(load_origination(columns=["loan_seq_num"], lazy=True).select(pl.len()).collect().item())
metrics["origination_total_all_vintages"] = orig_count

# Per-event_type breakdown (from outcomes table, in-sample years)
event_counts = (
    load_outcomes(years=YEARS, columns=["event_type"])
    .group_by("event_type").len()
    .to_pandas().set_index("event_type")["len"].to_dict()
)
metrics["event_type_counts"] = {str(k): int(v) for k, v in event_counts.items()}
total_outcomes = sum(metrics["event_type_counts"].values())
metrics["event_type_shares"] = {
    k: round(v / total_outcomes, 4) for k, v in metrics["event_type_counts"].items()
}
metrics["outcomes_total"] = int(total_outcomes)

# --- KM survival at key months -----------------------------------------------
S_km = {}
for t in (12, 24, 36, 60, 120, 180):
    avail = S_agg.index[S_agg.index <= t]
    if len(avail):
        S_km[t] = float(round(S_agg.loc[avail.max()], 4))
metrics["km_survival"] = S_km

# Median (first month where S drops below 0.5)
below_half = S_agg.index[S_agg.values < 0.5]
metrics["km_median_months"] = int(below_half.min()) if len(below_half) else None

# --- Aggregate hazard --------------------------------------------------------
disc = discrete_hazard[:EFFECTIVE_MAX_MONTHS]
sm   = smoothed[:EFFECTIVE_MAX_MONTHS]
metrics["effective_max_months"]     = int(EFFECTIVE_MAX_MONTHS)
metrics["peak_discrete_hazard_pct"] = float(round(disc.max() * 100, 3))
metrics["peak_discrete_hazard_month"] = int(disc.idxmax())
metrics["peak_smoothed_hazard_pct"] = float(round(sm.max() * 100, 3))
metrics["peak_smoothed_hazard_month"] = int(sm.idxmax())
# Plateau: mean smoothed hazard over months 30..120 (where it's roughly flat)
plateau = sm.loc[(sm.index >= 30) & (sm.index <= 120)]
metrics["plateau_smoothed_hazard_pct"] = {
    "min":    float(round(plateau.min()    * 100, 3)),
    "max":    float(round(plateau.max()    * 100, 3)),
    "mean":   float(round(plateau.mean()   * 100, 3)),
    "months": [30, 120],
}

# --- Log-rank summary --------------------------------------------------------
metrics["logrank_results"] = [
    {"covariate": r["covariate"],
     "chi2":      round(r["chi2"], 1),
     "p_value":   float(r["p_value"]),
     "n_groups":  int(r["n_groups"]),
     "n_loans":   int(r["n_loans"])}
    for r in sorted(LOGRANK_RESULTS, key=lambda x: -x["chi2"])
]

# --- Bin-free Cox stratification strength ------------------------------------
try:
    metrics["cox_binfree_strength"] = cox_strength.to_dict(orient="records")
except NameError:
    metrics["cox_binfree_strength_error"] = "cox_strength not defined — run bin-free Cox cell first"

# --- AJ vs KM gap at key months (re-derive cheap from existing variables) ----
# We saved aj_raw but deleted it. Recompute the gap by re-loading just enough.
try:
    aj_tmp = (
        load_outcomes(years=YEARS, columns=["event_time_months", "event_type"])
        .filter(pl.col("event_time_months").is_not_null())
        .to_pandas()
    )
    if len(aj_tmp) > 500_000:
        aj_tmp = aj_tmp.sample(500_000, random_state=0)
    from lifelines import AalenJohansenFitter
    event_map = {"censored": 0, "prepaid": 1, "defaulted": 2, "other_termination": 3}
    aj_tmp["event_code"] = aj_tmp["event_type"].map(event_map).fillna(0).astype(int)
    ajf = AalenJohansenFitter(calculate_variance=False)
    ajf.fit(aj_tmp["event_time_months"], aj_tmp["event_code"], event_of_interest=1)
    cif_prep = ajf.cumulative_density_.iloc[:, 0]

    cif_at = {}
    for t in (36, 60, 120, 180):
        idx = cif_prep.index[cif_prep.index <= t]
        if len(idx):
            cif_val = float(cif_prep.loc[idx.max()])
            km_val  = 1 - S_km.get(t, float("nan"))
            cif_at[t] = {
                "cif_prepay":     round(cif_val,        4),
                "km_one_minus_S": round(km_val,         4),
                "km_overstate":   round(km_val - cif_val, 4),
            }
    metrics["aj_vs_km_gap"] = cif_at
    del aj_tmp, ajf, cif_prep
except Exception as e:
    metrics["aj_vs_km_gap_error"] = str(e)

out_path = FIGURES_DIR / "partA_metrics.json"
out_path.write_text(json.dumps(metrics, indent=2, sort_keys=False))
print(f"wrote {out_path}")
print(json.dumps(metrics, indent=2, sort_keys=False))
"""


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # 1. Strip any previously inserted addendum cells (idempotency).
    nb["cells"] = [c for c in nb["cells"]
                   if not str(c.get("id", "")).startswith(MARKER)]

    # 2. Find insertion point: just before the final "Saved artifacts" markdown.
    insert_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "markdown":
            src = "".join(c.get("source", []))
            if "Saved artifacts" in src:
                insert_idx = i
                break
    if insert_idx is None:
        insert_idx = len(nb["cells"])  # fallback: append at end

    new_cells = [
        cell_md(OVERLAY_MD,           f"{MARKER}-overlay-md"),
        cell_code(OVERLAY_CODE,         f"{MARKER}-overlay-code"),
        cell_md(BAR_MD,               f"{MARKER}-bar-md"),
        cell_code(BAR_CODE,             f"{MARKER}-bar-code"),
        cell_md(DISCRETE_HAZARD_MD,   f"{MARKER}-disc-hazard-md"),
        cell_code(DISCRETE_HAZARD_CODE, f"{MARKER}-disc-hazard-code"),
        cell_md(KM_PER_COVARIATE_MD,  f"{MARKER}-km-covar-md"),
        cell_code(KM_PER_COVARIATE_CODE, f"{MARKER}-km-covar-code"),
        cell_md(COX_BINFREE_MD,       f"{MARKER}-cox-binfree-md"),
        cell_code(COX_BINFREE_CODE,     f"{MARKER}-cox-binfree-code"),
        cell_md(METRICS_MD,           f"{MARKER}-metrics-md"),
        cell_code(METRICS_CODE,         f"{MARKER}-metrics-code"),
    ]

    nb["cells"] = nb["cells"][:insert_idx] + new_cells + nb["cells"][insert_idx:]

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(f"updated {NB_PATH} — inserted {len(new_cells)} cells at index {insert_idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
