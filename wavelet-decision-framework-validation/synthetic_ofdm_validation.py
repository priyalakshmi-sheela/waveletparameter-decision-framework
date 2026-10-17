"""
synthetic_ofdm_validation.py - Case study 1: synthetic pilot-aided OFDM
channel-estimation benchmark.

Link: 128 subcarriers, 32-sample cyclic prefix, comb pilots on every 4th
subcarrier (32 pilots / 96 data), QPSK, 12-tap exponential power-delay-profile
Rayleigh channel renormalised per frame, fs = 30.72 MHz, B = 5 MHz,
SNR 0:5:20 dB, AWGN and bursty-impulsive noise, 80 frames per (regime, SNR),
40/60 validation/test split. The wavelet configuration is selected on the
validation split only; every reported number comes from the held-out test split.

Significance: paired Wilcoxon signed-rank test on per-frame BER, Holm-corrected
over the 8 comparisons against the framework's wavelet estimate.

Stages (each is a function, also used by stage_by_stage_validation.ipynb):
  1. simulate_dataset()          generate all frames
  2. split_frames()              stratified 40/60 validation/test split
  3. select_on_validation()      tune baselines and select the framework configuration
  4. evaluate_on_test()          per-frame receiver metrics of every method
  5. performance_table(), significance_table()

Outputs (results/):
  synthetic_ofdm_performance.csv    BER, EVM, NMSE, SNR gain per regime and method
  synthetic_ofdm_significance.csv   Holm-corrected p-values
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import decision_framework_core as core

N_SC, CP, PILOT_SPACING, CHANNEL_TAPS = 128, 32, 4, 12
FS_HZ, BW_HZ = 30.72e6, 5e6
SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
REGIMES = {"awgn": "AWGN", "bursty_impulsive": "Impulsive"}
FRAMES_PER_SNR, DATA_SEED, SPLIT_SEED, VAL_FRACTION = 80, 7, 10, 0.4
DD_ITERS, DD_GATE_DB = 2, 8.0
DFT_TAPS, CLIP_FACTORS = [4, 8, 12, 16, 24], [3.0, 4.0, 5.0]
SUPPORT_TAPS = [0, 4, 8, 12, 16, 24, 32]
FIXED = core.WaveletConfig("DWT", "db4", 2, "sqtwolog", "soft")        # fixed-parameter baseline
J_BAND = max(1, int(np.ceil(np.log2((FS_HZ / 2.0) / BW_HZ))))          # band-aligned depth = 2
LEVELS = list(range(max(1, J_BAND - 1), J_BAND + 2))                    # [1, 2, 3]
GRID = core.candidate_grid(N_SC, ["DWT", "SWT", "WPT"], ["haar", "db4", "sym4", "dmey"], LEVELS,
                           ["sqtwolog", "rigrsure"], ["soft", "garrote"], [0.5, 1.0],
                           ["component", "magnitude_phase"])
REPORTED = {"Raw LS": "LS_no_denoising", "DFT tuned": "LS_DFT_DD",
            "Fixed wavelet": "LS_fixed_wavelet_DD", "Proposed framework": "LS_framework_selected_wavelet_DD"}
OUT = Path(__file__).resolve().parent / "results"


# ------------------------------------------------------------ stage 1: data
def simulate_dataset():
    """All frames of both regimes, generated from one seeded random stream.
    Returns (FrameBatch, array of regime names per frame)."""
    rng = np.random.default_rng(DATA_SEED)
    pilot, data = core.pilot_data_indices(N_SC, PILOT_SPACING)
    pdp = np.exp(-np.arange(CHANNEL_TAPS) / max(CHANNEL_TAPS / 3.0, 1.0))
    pdp = pdp / np.sum(pdp)
    regime, snr, xs, ys, hs, bits_all = [], [], [], [], [], []
    for reg in REGIMES:
        for s in SNRS:
            for _ in range(FRAMES_PER_SNR):
                bits = rng.integers(0, 2, size=2 * len(data))
                x = np.zeros(N_SC, dtype=complex)
                x[pilot] = core.PILOT_SYMBOL
                x[data] = core.qpsk_mod(bits)
                x_t = np.fft.ifft(x)
                x_cp = np.concatenate([x_t[-CP:], x_t])
                h_t = (rng.normal(size=CHANNEL_TAPS) + 1j * rng.normal(size=CHANNEL_TAPS)) * np.sqrt(pdp / 2.0)
                h_t = h_t / np.sqrt(np.sum(np.abs(h_t) ** 2) + 1e-12)
                y_cp = core.add_noise(rng, np.convolve(x_cp, h_t)[:len(x_cp)], s, reg)
                regime.append(reg); snr.append(s); xs.append(x); bits_all.append(bits)
                ys.append(np.fft.fft(y_cp[CP:CP + N_SC])); hs.append(np.fft.fft(h_t, N_SC))
    fb = core.FrameBatch(np.arange(len(snr)), np.array(snr), np.array(xs), np.array(ys), np.array(hs),
                         np.array(bits_all), pilot, data)
    return fb, np.array(regime)


# ------------------------------------------------------------ stage 2: split
def split_frames(fb, regime_of):
    """Stratified 40/60 split. Returns {regime: (validation FrameBatch, test FrameBatch)}."""
    val_idx, test_idx = core.split_validation_test(fb.snr_db, regime_of, VAL_FRACTION, SPLIT_SEED)
    return {reg: (fb.subset(val_idx[regime_of[val_idx] == reg]), fb.subset(test_idx[regime_of[test_idx] == reg]))
            for reg in REGIMES}


# ------------------------------------------------------ stage 3: selection
def lexi_tune(val, estimator, candidates):
    """Tune a baseline parameter on the validation split (BER > EVM > NMSE)."""
    scores = [{k: float(np.mean(v)) for k, v in core.receiver_metrics(val, estimator(val.h_ls, c)).items()}
              for c in candidates]
    return candidates[core.lexicographic_best(scores)]


def select_on_validation(val):
    """Tune the DFT and clipping baselines and select the framework configuration,
    using the validation split only."""
    sel, scores = core.select_framework_config(val, GRID, SUPPORT_TAPS)
    return {"dft_taps": lexi_tune(val, core.dft_estimate, DFT_TAPS),
            "clip_factor": lexi_tune(val, core.clipping_estimate, CLIP_FACTORS),
            "framework": sel, "candidate_scores": scores}


# ---------------------------------------------------- stage 4: test split
def evaluate_on_test(val, test, selection):
    """Per-frame metrics (MMSE receiver) of every method on the held-out test split."""
    dft_k, clip_c, sel = selection["dft_taps"], selection["clip_factor"], selection["framework"]
    sel_k = sel.support_taps or dft_k
    h_fixed = core.wavelet_estimate(test.h_ls, FIXED)
    h_sel = core.wavelet_estimate(test.h_ls, sel)
    h_dft = core.dft_estimate(test.h_ls, dft_k)
    methods = {   # name: (channel estimate, support used for the MMSE noise estimate)
        "LS_no_denoising": (test.h_ls, None),
        "LS_DFT_validation_tuned": (h_dft, dft_k),
        "LS_LMMSE_covariance": (core.lmmse_estimate(val, test.h_ls), None),
        "LS_delay_MAD_clipping": (core.clipping_estimate(test.h_ls, clip_c), None),
        "LS_fixed_wavelet": (h_fixed, None),
        "LS_framework_selected_wavelet": (h_sel, sel.support_taps or None),
        "LS_DFT_DD": (core.decision_directed(test, h_dft, dft_k, DD_ITERS, DD_GATE_DB), dft_k),
        "LS_fixed_wavelet_DD": (core.decision_directed(test, h_fixed, dft_k, DD_ITERS, DD_GATE_DB), None),
        "LS_framework_selected_wavelet_DD": (core.decision_directed(test, h_sel, sel_k, DD_ITERS, DD_GATE_DB),
                                             sel.support_taps or None),
    }
    return {name: core.receiver_metrics(test, h, "mmse", k) for name, (h, k) in methods.items()}


# ------------------------------------------------------ stage 5: tables
def holm(p):
    """Holm step-down adjustment."""
    p = np.asarray(p, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def performance_table(label, metrics):
    """Rows of the performance table for one regime (test-split means)."""
    return [{"Noise Regime": label, "Method": row,
             "BER": round(float(np.mean(metrics[name]["ber"])), 4),
             "EVM (%)": round(float(np.mean(metrics[name]["evm_percent"])), 1),
             "NMSE": round(float(np.mean(metrics[name]["nmse"])), 4),
             "SNR Gain (dB)": round(float(np.mean(metrics[name]["snr_gain_db"])), 2)}
            for row, name in REPORTED.items()]


def significance_table(label, metrics):
    """Holm-corrected Wilcoxon p-values (per-frame BER) for one regime."""
    ref = "LS_framework_selected_wavelet"
    others = [n for n in sorted(metrics) if n != ref]
    adj = dict(zip(others, holm([wilcoxon(metrics[n]["ber"], metrics[ref]["ber"]).pvalue for n in others])))
    return [{"Noise Regime": label, "Comparison": "framework vs fixed wavelet",
             "p (Holm)": f"{adj['LS_fixed_wavelet']:.1e}"},
            {"Noise Regime": label, "Comparison": "framework vs raw LS",
             "p (Holm)": f"{adj['LS_no_denoising']:.1e}"}]


def main():
    OUT.mkdir(exist_ok=True)
    fb, regime_of = simulate_dataset()
    splits = split_frames(fb, regime_of)
    performance, significance = [], []
    for reg, label in REGIMES.items():
        val, test = splits[reg]
        selection = select_on_validation(val)
        print(f"[{label}] selected framework configuration: {selection['framework'].label()}")
        metrics = evaluate_on_test(val, test, selection)
        performance += performance_table(label, metrics)
        significance += significance_table(label, metrics)

    performance, significance = pd.DataFrame(performance), pd.DataFrame(significance)
    performance.to_csv(OUT / "synthetic_ofdm_performance.csv", index=False)
    significance.to_csv(OUT / "synthetic_ofdm_significance.csv", index=False)
    print("\nSynthetic OFDM link - held-out test performance\n")
    print(performance.to_string(index=False, formatters={
        "BER": "{:.4f}".format, "EVM (%)": "{:.1f}".format,
        "NMSE": "{:.4f}".format, "SNR Gain (dB)": "{:.2f}".format}))
    print("\nHolm-corrected Wilcoxon signed-rank test (per-frame BER)\n")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()
