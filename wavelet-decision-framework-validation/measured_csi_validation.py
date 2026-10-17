"""
measured_csi_validation.py - Case study 2: measured IEEE 802.11n channels.

Data: data/measured_80211n_csi_pool.npy - 445 measured channel-state-
information vectors x 56 subcarriers (Atheros 802.11n). Each frame draws one
measured channel, applies it to a comb-pilot QPSK OFDM symbol and adds AWGN or
bursty-impulsive noise; 80 frames per (regime, SNR), SNR 0:5:20 dB, 40/60
validation/test split. The wavelet configuration is selected on the
validation split only; every reported number comes from the held-out test split.

Significance: paired Wilcoxon signed-rank test on per-frame channel NMSE of
the AWGN test frames (wavelet estimates before decision-directed refinement).

Stages (each is a function, also used by stage_by_stage_validation.ipynb):
  1. load_pool()                 measured channels, unit average power
  2. build_frames(), split_frames()
  3. select_on_validation()      tune the DFT baseline and select the framework configuration
  4. evaluate_on_test()          per-frame receiver metrics on the held-out test split
  5. performance_table(), significance_table()

Outputs (results/):
  measured_csi_performance.csv    BER, EVM, NMSE, SNR gain per regime and method
  measured_csi_significance.csv   p-values
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import decision_framework_core as core

HERE = Path(__file__).resolve().parent
CSI_POOL = HERE / "data" / "measured_80211n_csi_pool.npy"
OUT = HERE / "results"
N_SC, PILOT_SPACING = 56, 4
SNRS = [0, 5, 10, 15, 20]
FRAMES_PER_SNR, SPLIT_SEED, VAL_FRACTION = 80, 10, 0.4
REGIMES = {"awgn": ("AWGN", 7), "bursty_impulsive": ("Impulsive", 11)}   # label, data seed
DD_ITERS, DD_GATE_DB = 2, 8.0
DFT_TAPS = [4, 8, 14, 20, 28, 56]
SUPPORT_TAPS = [0, 4, 8, 14, 20, 28, 56]
FIXED = core.WaveletConfig("DWT", "db4", 2, "sqtwolog", "soft")        # fixed-parameter baseline
GRID = core.candidate_grid(N_SC, ["DWT", "SWT", "WPT"], ["haar", "db4", "sym4"], [1, 2, 3],
                           ["sqtwolog", "rigrsure"], ["soft", "garrote"], [0.5, 1.0], ["component"])
REPORTED = ["Raw LS", "DFT tuned", "Fixed wavelet", "Proposed framework"]


# ------------------------------------------------------- stages 1-2: data
def load_pool():
    """Measured channel pool, each channel normalised to unit average power."""
    pool = np.load(CSI_POOL)
    return pool / np.sqrt((np.abs(pool) ** 2).mean(1, keepdims=True))


def build_frames(pool, regime, seed):
    """400 trace-driven frames for one noise regime from one seeded random stream."""
    rng = np.random.default_rng(seed)
    pilot, data = core.pilot_data_indices(N_SC, PILOT_SPACING)
    xs, ys, hs, bits_all, snrs = [], [], [], [], []
    for snr in SNRS:
        for _ in range(FRAMES_PER_SNR):
            h = pool[rng.integers(len(pool))]
            bits = rng.integers(0, 2, 2 * len(data))
            x = np.zeros(N_SC, dtype=complex)
            x[pilot] = core.PILOT_SYMBOL
            x[data] = core.qpsk_mod(bits)
            ys.append(np.fft.fft(core.add_noise(rng, np.fft.ifft(h * x), snr, regime)))
            xs.append(x); hs.append(h.copy()); bits_all.append(bits); snrs.append(float(snr))
    return core.FrameBatch(np.arange(len(snrs)), np.array(snrs), np.array(xs), np.array(ys),
                           np.array(hs), np.array(bits_all), pilot, data)


def split_frames(fb, regime):
    """Stratified 40/60 split. Returns (validation FrameBatch, test FrameBatch)."""
    val_idx, test_idx = core.split_validation_test(fb.snr_db, np.array([regime] * len(fb.snr_db)),
                                                   VAL_FRACTION, SPLIT_SEED)
    return fb.subset(val_idx), fb.subset(test_idx)


# ------------------------------------------------------ stage 3: selection
def select_on_validation(val):
    """Select the framework configuration and tune the DFT baseline (by NMSE),
    using the validation split only."""
    sel, scores = core.select_framework_config(val, GRID, SUPPORT_TAPS)
    dft_nmse = [float(np.mean(core.plain_nmse(val, core.dft_estimate(val.h_ls, k), eps_den=0.0)))
                for k in DFT_TAPS]
    return {"dft_taps": DFT_TAPS[int(np.argmin(dft_nmse))], "framework": sel, "candidate_scores": scores}


# ---------------------------------------------------- stage 4: test split
def evaluate_on_test(test, selection):
    """Per-frame metrics (MMSE receiver) on the held-out test split.
    'Fixed wavelet (no DD)' and 'Proposed framework (no DD)' are the estimates
    before decision-directed refinement, used by the significance test."""
    dft_k, sel = selection["dft_taps"], selection["framework"]
    sel_k = sel.support_taps or dft_k
    h_fixed = core.wavelet_estimate(test.h_ls, FIXED)
    h_sel = core.wavelet_estimate(test.h_ls, sel)
    methods = {   # name: (channel estimate, support used for the MMSE noise estimate)
        "Raw LS": (test.h_ls, None),
        "DFT tuned": (core.decision_directed(test, core.dft_estimate(test.h_ls, dft_k), dft_k,
                                             DD_ITERS, DD_GATE_DB), dft_k),
        "Fixed wavelet": (core.decision_directed(test, h_fixed, dft_k, DD_ITERS, DD_GATE_DB), None),
        "Proposed framework": (core.decision_directed(test, h_sel, sel_k, DD_ITERS, DD_GATE_DB),
                               sel.support_taps or None),
        "Fixed wavelet (no DD)": (h_fixed, None),
        "Proposed framework (no DD)": (h_sel, sel.support_taps or None),
    }
    return {name: core.receiver_metrics(test, h, "mmse", k) for name, (h, k) in methods.items()}


# ------------------------------------------------------ stage 5: tables
def performance_table(label, metrics):
    """Rows of the performance table for one regime (test-split means)."""
    return [{"Noise Regime": label, "Method": row,
             "BER": round(float(np.mean(metrics[row]["ber"])), 4),
             "EVM (%)": round(float(np.mean(metrics[row]["evm_percent"])), 1),
             "NMSE": round(float(np.mean(metrics[row]["nmse"])), 4),
             "SNR Gain (dB)": round(float(np.mean(metrics[row]["snr_gain_db"])), 2)}
            for row in REPORTED]


def significance_table(metrics):
    """Wilcoxon p-values on per-frame channel NMSE (AWGN regime metrics)."""
    fw = metrics["Proposed framework (no DD)"]["nmse"]
    return [{"Comparison": f"framework vs {label}",
             "p": f"{wilcoxon(metrics[name]['nmse'], fw).pvalue:.1e}"}
            for label, name in [("raw LS", "Raw LS"), ("fixed wavelet", "Fixed wavelet (no DD)")]]


def main():
    OUT.mkdir(exist_ok=True)
    pool = load_pool()
    performance, significance = [], []
    for reg, (label, seed) in REGIMES.items():
        val, test = split_frames(build_frames(pool, reg, seed), reg)
        selection = select_on_validation(val)
        print(f"[{label}] selected framework configuration: {selection['framework'].label()}")
        metrics = evaluate_on_test(test, selection)
        performance += performance_table(label, metrics)
        if reg == "awgn":
            significance += significance_table(metrics)

    performance, significance = pd.DataFrame(performance), pd.DataFrame(significance)
    performance.to_csv(OUT / "measured_csi_performance.csv", index=False)
    significance.to_csv(OUT / "measured_csi_significance.csv", index=False)
    print("\nMeasured IEEE 802.11n channels - held-out test performance\n")
    print(performance.to_string(index=False, formatters={
        "BER": "{:.4f}".format, "EVM (%)": "{:.1f}".format,
        "NMSE": "{:.4f}".format, "SNR Gain (dB)": "{:.2f}".format}))
    print("\nWilcoxon signed-rank test (per-frame channel NMSE, AWGN)\n")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()
