"""
transform_family_benchmark.py - DWT vs SWT vs WPT impulse suppression in an
OFDM receiver under Middleton Class-A noise.

Link: 64-point OFDM, 16-sample cyclic prefix, 48 active QPSK subcarriers,
14 OFDM symbols per frame (1120 complex samples), unit-power waveform.
Operating points: Eb/N0 in {6, 10, 14} dB x Middleton A in {0.005, 0.02} x
T in {0.05, 0.20} x 5 seeds = 60 matched operating points of 16 frames each.

For each transform family, every configuration of the search space
  {db4, sym4, coif4} x J in {2, 3} x {scaled MAD (2.5 sigma), universal, BayesShrink}
  x {soft, hard, garrote}   (54 per family)
is applied to all frames, and the configuration with the largest mean
denoising gain (complex SNR gain) is selected. The selected configurations are
then compared at the 60 matched operating points.

Stages (each is a function, also used by stage_by_stage_benchmark.ipynb):
  1. simulate_operating_points()   frames of the 60 operating points
  2. search_configurations()       evaluate the search space of each family
  3. select_configurations()       largest mean denoising gain per family
  4. measure_latency()             per-frame processing time of the selected configurations
  5. performance_table()           Table 6
     complexity_table()            Table 7
     friedman_table()              Table 8
     posthoc_table()               pairwise Wilcoxon tests (Holm) quoted in the text

Outputs (results/):
  transform_performance.csv         Table 6 (mean and 95 % CI half-width)
  transform_complexity.csv          Table 7
  transform_friedman.csv            Table 8
  transform_posthoc.csv             pairwise Wilcoxon signed-rank tests, Holm-corrected
  transform_operating_points.csv    per-operating-point metrics of the selected configurations
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

import transform_benchmark_core as core

OUT = Path(__file__).resolve().parent / "results"
EBN0_DB, A_VALUES, T_VALUES = (6.0, 10.0, 14.0), (0.005, 0.02), (0.05, 0.20)
SEEDS = (101, 202, 303, 404, 505)
FRAMES_PER_POINT, SYMBOLS_PER_FRAME = 16, 14
TRANSFORMS = ("DWT", "SWT", "WPT")
WAVELETS, LEVELS = ("db4", "sym4", "coif4"), (2, 3)
THRESHOLD_FUNCTIONS, THRESHOLD_RULES, MAD_SCALE = ("scaled_mad", "universal", "bayes"), ("soft", "hard", "garrote"), 2.5
TIMING_REPEATS, WARMUP_CALLS = 3, 3          # latency = median of 3 timed calls per frame, after 3 warm-up calls
METRICS = {"gain_db": "ΔSNR", "nmse": "MSE/NMSE", "ber": "BER", "evm_percent": "EVM", "time_ms": "Mean time"}


# ------------------------------------------------------------ stage 1: data
@dataclass
class OperatingPoint:
    seed: int
    ebn0_db: float
    A: float
    T: float
    tx: np.ndarray            # (frames, samples) transmitted waveform, unit average power
    rx: np.ndarray            # (frames, samples) received waveform
    bits: np.ndarray          # (frames, bits)
    snr_in_db: np.ndarray     # (frames,)


def simulate_operating_points():
    """All frames of the 60 operating points; each operating point has its own seeded stream."""
    n_bits = 2 * len(core.ACTIVE_BINS) * SYMBOLS_PER_FRAME
    points = []
    for ebn0 in EBN0_DB:
        for A in A_VALUES:
            for T in T_VALUES:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    tx_all, rx_all, bits_all, snr_in = [], [], [], []
                    for _ in range(FRAMES_PER_POINT):
                        bits = rng.integers(0, 2, n_bits)
                        tx = core.ofdm_modulate(bits)
                        tx = tx / np.sqrt(np.mean(np.abs(tx) ** 2))
                        sigma2 = (np.sum(np.abs(tx) ** 2) / len(bits)) / (10 ** (ebn0 / 10))   # Eb / (Eb/N0)
                        rx = tx + core.middleton_class_a(rng, len(tx), sigma2, A, T)
                        tx_all.append(tx); rx_all.append(rx); bits_all.append(bits)
                        snr_in.append(core.snr_db(tx, rx))
                    points.append(OperatingPoint(seed, ebn0, A, T, np.array(tx_all), np.array(rx_all),
                                                 np.array(bits_all), np.array(snr_in)))
    return points


# --------------------------------------------------- stage 2: search space
def candidate_grid(transform):
    return [core.TransformConfig(transform, w, lvl, fn, rule, MAD_SCALE)
            for w in WAVELETS for lvl in LEVELS for fn in THRESHOLD_FUNCTIONS for rule in THRESHOLD_RULES]


def evaluate_accuracy(points, cfg):
    """Per-operating-point denoising gain, NMSE, BER and EVM (frame averages) of one configuration."""
    denoised = core.suppress_impulses(np.concatenate([p.rx for p in points]), cfg)
    symbols = core.ofdm_demodulate(denoised)
    out = {k: [] for k in ("gain_db", "nmse", "ber", "evm_percent")}
    row = 0
    for p in points:
        snr_in = snr_out = nmse = evm = 0.0
        errors = 0
        for f in range(len(p.tx)):
            ref = core.qpsk_mod(p.bits[f])
            errors += int(np.sum(core.qpsk_demod(symbols[row]) != p.bits[f]))
            snr_in += p.snr_in_db[f]
            snr_out += core.snr_db(p.tx[f], denoised[row])
            nmse += core.nmse(p.tx[f], denoised[row])
            evm += core.evm_rms_percent(ref, symbols[row])
            row += 1
        n = len(p.tx)
        out["gain_db"].append(snr_out / n - snr_in / n)
        out["nmse"].append(nmse / n)
        out["ber"].append(errors / p.bits.size)
        out["evm_percent"].append(evm / n)
    return {k: np.array(v) for k, v in out.items()}


def search_configurations(points):
    """Evaluate every admissible configuration of every family.
    Returns {transform: list of (config, per-operating-point metrics)}."""
    n = points[0].tx.shape[1]
    return {t: [(cfg, evaluate_accuracy(points, cfg)) for cfg in candidate_grid(t) if core.level_supported(cfg, n)]
            for t in TRANSFORMS}


# ------------------------------------------------------- stage 3: selection
def select_configurations(search, n_samples):
    """Largest mean denoising gain per family (ties: lower FLOP proxy, then grid order).
    Returns {transform: (config, per-operating-point metrics)}."""
    selected = {}
    for t, results in search.items():
        best = max(range(len(results)), key=lambda i: (float(np.mean(results[i][1]["gain_db"])),
                                                       -core.flop_proxy(results[i][0], n_samples), -i))
        selected[t] = results[best]
    return selected


def candidate_ranking(search, transform):
    """Mean metrics of every candidate of one family, best first."""
    df = pd.DataFrame([{"configuration": cfg.label(), "ΔSNR (dB)": float(np.mean(m["gain_db"])),
                        "NMSE": float(np.mean(m["nmse"])), "BER": float(np.mean(m["ber"])),
                        "EVM (%)": float(np.mean(m["evm_percent"]))} for cfg, m in search[transform]])
    df = df.sort_values("ΔSNR (dB)", ascending=False, kind="mergesort").reset_index(drop=True)
    df.index += 1
    return df.rename_axis("rank")


# -------------------------------------------------------- stage 4: latency
def measure_latency(points, selected):
    """Per-operating-point mean processing time (ms/frame) of each selected configuration.
    Each frame is processed separately; the time of a frame is the median of
    TIMING_REPEATS calls. This is a wall-clock measurement and depends on the computer."""
    latency = {}
    for t, (cfg, _) in selected.items():
        for _ in range(WARMUP_CALLS):
            core.suppress_impulses(points[0].rx[0], cfg)
        per_point = []
        for p in points:
            total = 0.0
            for frame in p.rx:
                calls = []
                for _ in range(TIMING_REPEATS):
                    t0 = time.perf_counter()
                    core.suppress_impulses(frame, cfg)
                    calls.append(time.perf_counter() - t0)
                total += float(np.median(calls))
            per_point.append(total / len(p.rx) * 1000.0)
        latency[t] = np.array(per_point)
    return latency


def operating_point_table(points, selected, latency):
    """Matched per-operating-point observations (the paired data of Table 8)."""
    rows = []
    for t, (cfg, m) in selected.items():
        for i, p in enumerate(points):
            rows.append({"transform": t, "configuration": cfg.label(), "seed": p.seed, "ebn0_db": p.ebn0_db,
                         "A": p.A, "T": p.T, "gain_db": m["gain_db"][i], "nmse": m["nmse"][i],
                         "ber": m["ber"][i], "evm_percent": m["evm_percent"][i], "time_ms": latency[t][i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- stage 5: tables
def _pm(mean, ci, digits=3):
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def _sci_pm(mean, ci, digits=3):
    e = int(np.floor(np.log10(abs(mean))))
    return f"({mean / 10 ** e:.{digits}f} ± {ci / 10 ** e:.{digits}f}) × 10^{e}"


def _sci(p):
    m, e = f"{p:.2e}".split("e")
    return f"{m} × 10^{int(e)}"


def _paired(op_table, metric):
    return [op_table.loc[op_table["transform"] == t, metric].to_numpy() for t in TRANSFORMS]


def performance_table(op_table):
    """Table 6: mean ± 95 % CI over the 60 operating points. Returns (numeric, formatted)."""
    num, fmt = [], []
    for t, g in op_table.groupby("transform", sort=False):
        r = {"Method": t, "Selected configuration": g["configuration"].iloc[0]}
        for col in ("gain_db", "nmse", "ber", "evm_percent"):
            r[col], r[col + "_ci95"] = float(g[col].mean()), core.ci95(g[col])
        num.append(r)
        fmt.append({"Method": t, "ΔSNR / denoising gain": _pm(r["gain_db"], r["gain_db_ci95"]) + " dB",
                    "NMSE": _sci_pm(r["nmse"], r["nmse_ci95"]), "BER outcome": _sci_pm(r["ber"], r["ber_ci95"]),
                    "EVM": _pm(r["evm_percent"], r["evm_percent_ci95"]) + " %"})
    return pd.DataFrame(num), pd.DataFrame(fmt)


def complexity_table(op_table, selected, n_samples):
    """Table 7: mean time ± 95 % CI, FLOP proxy and minimum frame budget (= mean time)."""
    num, fmt = [], []
    for t, g in op_table.groupby("transform", sort=False):
        mean, ci, flops = float(g["time_ms"].mean()), core.ci95(g["time_ms"]), core.flop_proxy(selected[t][0], n_samples)
        num.append({"Method": t, "time_ms": mean, "time_ms_ci95": ci, "flop_proxy": flops,
                    "minimum_frame_budget_ms": mean})
        fmt.append({"Method": t, "Mean time": _pm(mean, ci) + " ms/frame", "FLOP proxy": f"{flops:,.0f}",
                    "Minimum frame budget Tframe": f"{mean:.3f} ms"})
    return pd.DataFrame(num), pd.DataFrame(fmt)


def friedman_table(op_table):
    """Table 8: Friedman test across DWT, SWT and WPT at the matched operating points."""
    num, fmt = [], []
    for col, label in METRICS.items():
        chi2, p = friedmanchisquare(*_paired(op_table, col))
        num.append({"Metric": label, "n": int((op_table["transform"] == TRANSFORMS[0]).sum()),
                    "chi2": float(chi2), "p": float(p)})
        fmt.append({"Metric": label, "Friedman result": f"χ²(2) = {chi2:.3f}, p = {_sci(p)}"})
    return pd.DataFrame(num), pd.DataFrame(fmt)


def holm(p):
    """Holm step-down adjustment."""
    p = np.asarray(p, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def posthoc_table(op_table):
    """Pairwise Wilcoxon signed-rank tests, Holm-corrected over the 3 pairs of each metric."""
    rows = []
    for col, label in METRICS.items():
        x = dict(zip(TRANSFORMS, _paired(op_table, col)))
        pairs = [("DWT", "SWT"), ("DWT", "WPT"), ("SWT", "WPT")]
        tests = [wilcoxon(x[a], x[b], zero_method="wilcox") for a, b in pairs]
        for (a, b), res, p_adj in zip(pairs, tests, holm([r.pvalue for r in tests])):
            rows.append({"Metric": label, "Comparison": f"{a} vs {b}",
                         "Mean difference": float(np.mean(x[a] - x[b])),
                         "Wilcoxon statistic": float(res.statistic), "p (Holm)": float(p_adj)})
    return pd.DataFrame(rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # consoles without Unicode support show '?' instead of failing
    OUT.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    print("Simulating 60 operating points ...")
    points = simulate_operating_points()
    n_samples = points[0].tx.shape[1]
    print("Searching 54 configurations per transform family ...")
    selected = select_configurations(search_configurations(points), n_samples)
    for t, (cfg, _) in selected.items():
        print(f"[{t}] selected configuration: {cfg.label()}")
    print("Measuring per-frame processing time of the selected configurations ...")
    op_table = operating_point_table(points, selected, measure_latency(points, selected))

    perf, perf_fmt = performance_table(op_table)
    comp, comp_fmt = complexity_table(op_table, selected, n_samples)
    fried, fried_fmt = friedman_table(op_table)
    posthoc = posthoc_table(op_table)
    perf.to_csv(OUT / "transform_performance.csv", index=False)
    comp.to_csv(OUT / "transform_complexity.csv", index=False)
    fried.to_csv(OUT / "transform_friedman.csv", index=False)
    posthoc.to_csv(OUT / "transform_posthoc.csv", index=False)
    op_table.to_csv(OUT / "transform_operating_points.csv", index=False)

    print("\nReceiver-level denoising performance (Table 6)\n")
    print(perf_fmt.to_string(index=False))
    print("\nComputational complexity (Table 7)\n")
    print(comp_fmt.to_string(index=False))
    print("Time is measured on this computer; the article reports 0.171, 0.382 and 0.266 ms/frame "
          "measured on the authors' PC.")
    print("\nFriedman tests across DWT, SWT and WPT, 60 matched operating points (Table 8)\n")
    print(fried_fmt.to_string(index=False))
    print("\nPairwise Wilcoxon signed-rank tests, Holm-corrected (p values)\n")
    print(posthoc.pivot(index="Comparison", columns="Metric", values="p (Holm)")[list(METRICS.values())]
          .apply(lambda c: c.map("{:.1e}".format)).to_string())
    print(f"\nFinished in {time.perf_counter() - t_start:.0f} s. Results are in {OUT}")


if __name__ == "__main__":
    main()
