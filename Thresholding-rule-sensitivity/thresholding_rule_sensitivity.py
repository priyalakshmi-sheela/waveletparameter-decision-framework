"""
thresholding_rule_sensitivity.py - BER of an OFDM/QPSK receiver with wavelet
impulse suppression under AWGN and Middleton Class-A impulsive noise, for the
hard, soft and garrote thresholding rules (Fig. 12 and Supplementary Figs. S1
and S2 of the article).

Link: 64-point OFDM, 16-sample cyclic prefix, 48 active QPSK subcarriers,
14 OFDM symbols per frame (1120 complex samples, 1344 bits), unit-power waveform.
Noise: complex AWGN and complex Middleton Class A (A = 0.005, T = 0.05), with the
total noise power set by Eb/N0 in {0, 2, ..., 16} dB; 120 frames per Eb/N0 value.
Both noise models are applied to the same transmitted frames.
Receiver: wavelet impulse suppression (threshold 4 x MAD noise estimate of each
detail level, periodization) for every combination of
  mother wavelet psi in {haar, db4, sym4, coif4, bior4.4}
  decomposition level J in {2, 3, 4, 5}
  thresholding rule in {hard, soft, garrote}
followed by OFDM demodulation and hard QPSK decisions.

Fig. 12 shows the db4, J = 3 configuration; Supplementary Figs. S1 (AWGN) and
S2 (impulsive noise) show the whole 5 x 4 grid. All three figures come from the
same simulation.

Stages (each is a function, also used by stage_by_stage_thresholding.ipynb):
  1. simulate_frames()          transmitted bits and the AWGN / impulsive-noise received frames
  2. evaluate_ber()             BER of every configuration and of the receiver without denoising
  3. fig12_table(),
     plot_fig12_panel()         Fig. 12 (a) AWGN and (b) impulsive noise (db4, J = 3)
  4. plot_sensitivity_grid()    Supplementary Fig. S1 (AWGN) and Fig. S2 (impulsive noise)

Outputs (results/):
  fig12a_thresholding_awgn.pdf / .png                 Fig. 12(a)
  fig12b_thresholding_impulsive.pdf / .png            Fig. 12(b)
  figS1_thresholding_sensitivity_awgn.pdf / .png      Supplementary Fig. S1
  figS2_thresholding_sensitivity_impulsive.pdf / .png Supplementary Fig. S2
  fig12_ber_db4_J3.csv                                BER values plotted in Fig. 12
  thresholding_ber_all_configurations.csv             bit errors and BER of every configuration (S1, S2)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import thresholding_core as core

OUT = Path(__file__).resolve().parent / "results"
EBN0_DB = tuple(range(0, 17, 2))               # 0, 2, ..., 16 dB
N_FRAMES, SYMBOLS_PER_FRAME, SEED = 120, 14, 21
A, T = 0.005, 0.05                             # Middleton Class-A impulsive index and power ratio
THRESHOLD_SCALE = 4.0                          # threshold = 4 x MAD noise estimate of each detail level
WAVELETS, LEVELS = ("haar", "db4", "sym4", "coif4", "bior4.4"), (2, 3, 4, 5)
FIG12_WAVELET, FIG12_LEVEL = "db4", 3
NOISES = ("AWGN", "Middleton")
NOISE_TITLE = {"AWGN": "AWGN", "Middleton": "Impulsive Noise"}
CURVES = ("No denoising", "Hard", "Soft", "Garrote")
MARKERS = {"No denoising": "o", "Hard": "s", "Soft": "^", "Garrote": "D"}
FIG12_FILES = {"AWGN": "fig12a_thresholding_awgn", "Middleton": "fig12b_thresholding_impulsive"}
GRID_FILES = {"AWGN": "figS1_thresholding_sensitivity_awgn", "Middleton": "figS2_thresholding_sensitivity_impulsive"}
DPI = 600


# ------------------------------------------------------------ stage 1: frames
@dataclass
class EbN0Frames:
    ebn0_db: float
    bits: np.ndarray          # (frames, bits) transmitted bits
    rx: dict                  # noise model -> (frames, samples) received waveform


def simulate_frames(ebn0_db_values=EBN0_DB, n_frames=N_FRAMES, seed=SEED):
    """Transmitted bits and received frames for every Eb/N0 value. One random stream
    (seeded once) is used for everything, in the order: bits, AWGN, Middleton noise
    of frame 1, then frame 2, ...; both noise models are added to the same frame."""
    rng = np.random.default_rng(seed)
    n_bits = 2 * len(core.ACTIVE_BINS) * SYMBOLS_PER_FRAME
    frames = []
    for ebn0 in ebn0_db_values:
        bits_all, rx_awgn, rx_mid = [], [], []
        for _ in range(n_frames):
            bits = rng.integers(0, 2, n_bits)
            tx = core.ofdm_modulate(bits)
            tx = tx / np.sqrt(np.mean(np.abs(tx) ** 2))
            eb = np.sum(np.abs(tx) ** 2) / len(bits)
            sigma2 = eb / (10 ** (float(ebn0) / 10))
            rx_awgn.append(tx + core.awgn(rng, len(tx), sigma2))
            rx_mid.append(tx + core.middleton_class_a(rng, len(tx), sigma2, A, T))
            bits_all.append(bits)
        frames.append(EbN0Frames(float(ebn0), np.array(bits_all),
                                 {"AWGN": np.array(rx_awgn), "Middleton": np.array(rx_mid)}))
    return frames


# --------------------------------------------------------------- stage 2: BER
def _bit_errors(rx, bits):
    return int(np.sum(core.qpsk_demod(core.ofdm_demodulate(rx)) != bits))


def evaluate_ber(frames, wavelets=WAVELETS, levels=LEVELS, rules=core.THRESHOLD_RULES,
                 threshold_scale=THRESHOLD_SCALE, progress=True):
    """Bit errors and BER of every (wavelet, level, rule) and of the receiver without
    denoising (repeated in every wavelet/level block), for both noise models."""
    n_samples = frames[0].rx["AWGN"].shape[1]
    bad = [(w, lvl) for w in wavelets for lvl in levels if not core.level_supported(w, lvl, n_samples)]
    if bad:
        raise ValueError(f"decomposition level too deep for {n_samples}-sample frames: {bad}")
    rows = []
    for fr in frames:
        total = fr.bits.size
        baseline = {noise: _bit_errors(fr.rx[noise], fr.bits) for noise in NOISES}
        for w in wavelets:
            for lvl in levels:
                for noise in NOISES:
                    rows.append({"noise": noise, "wavelet": w, "level": lvl, "EbN0_dB": fr.ebn0_db,
                                 "threshold_rule": "No denoising", "bit_errors": baseline[noise],
                                 "total_bits": total, "BER": baseline[noise] / total})
                    for rule in rules:
                        denoised = core.suppress_impulses(fr.rx[noise], w, lvl, rule, threshold_scale)
                        err = _bit_errors(denoised, fr.bits)
                        rows.append({"noise": noise, "wavelet": w, "level": lvl, "EbN0_dB": fr.ebn0_db,
                                     "threshold_rule": rule.capitalize(), "bit_errors": err,
                                     "total_bits": total, "BER": err / total})
        if progress:
            print(f"  Eb/N0 = {fr.ebn0_db:g} dB done", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ stage 3: Fig. 12
def fig12_table(ber, wavelet=FIG12_WAVELET, level=FIG12_LEVEL):
    """BER values plotted in Fig. 12: one row per Eb/N0, one column per noise model and curve."""
    sub = ber[(ber["wavelet"] == wavelet) & (ber["level"] == level)]
    table = pd.DataFrame({"EbN0_dB": sorted(sub["EbN0_dB"].unique())})
    for noise in NOISES:
        for curve in CURVES:
            s = sub[(sub["noise"] == noise) & (sub["threshold_rule"] == curve)].set_index("EbN0_dB")["BER"]
            table[f"{noise} - {curve}"] = table["EbN0_dB"].map(s)
    return table


def plot_fig12_panel(table, noise):
    """One panel of Fig. 12: BER versus Eb/N0 for no denoising and the three rules."""
    fig, ax = plt.subplots(1, 1, figsize=(3.55, 3.65), constrained_layout=False)
    for curve in CURVES:
        ax.semilogy(table["EbN0_dB"], table[f"{noise} - {curve}"], linewidth=1.5, marker=MARKERS[curve],
                    markersize=4.8, label=curve)
    ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=8.5)
    ax.tick_params(axis="both", which="minor", labelsize=8.5)
    ax.grid(True, which="both", alpha=0.28)
    leg = ax.legend(fontsize=8.1, loc="lower left", frameon=True, framealpha=0.95)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
    return fig


# ---------------------------------------------------- stage 4: Figs. S1 and S2
def plot_sensitivity_grid(ber, noise, wavelets=WAVELETS, levels=LEVELS):
    """Supplementary Fig. S1 (AWGN) or S2 (impulsive noise): rows are mother wavelets,
    columns are decomposition levels; every panel shows the four BER curves."""
    fig, axes = plt.subplots(len(wavelets), len(levels), figsize=(9.2, 10.6), sharex=True, sharey=True,
                             constrained_layout=False, squeeze=False)
    for r, w in enumerate(wavelets):
        for c, lvl in enumerate(levels):
            ax = axes[r, c]
            sub = ber[(ber["noise"] == noise) & (ber["wavelet"] == w) & (ber["level"] == lvl)]
            for curve in CURVES:
                s = sub[sub["threshold_rule"] == curve].sort_values("EbN0_dB")
                ax.semilogy(s["EbN0_dB"], s["BER"], linewidth=1.2, marker=MARKERS[curve], markersize=3.2,
                            label=curve)
            ax.set_title(f"{w}, $J={lvl}$", fontsize=8.2, fontweight="bold", pad=2.0)
            ax.tick_params(axis="both", which="major", labelsize=7.0)
            ax.tick_params(axis="both", which="minor", labelsize=7.0)
            ax.grid(True, which="both", alpha=0.28)
            if r == len(wavelets) - 1:
                ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=8.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel("BER", fontsize=8.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=len(CURVES), frameon=True, framealpha=0.95,
                     fontsize=8.2, bbox_to_anchor=(0.5, 0.962))
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.suptitle(f"Thresholding sensitivity over wavelet family and decomposition level: {NOISE_TITLE[noise]}",
                 fontsize=10.5, fontweight="bold", y=0.992)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.055, top=0.925, hspace=0.35, wspace=0.12)
    return fig


def save_figure(fig, stem, out_dir=OUT, dpi=DPI):
    """Save a figure as PDF and as PNG (600 dpi). Returns the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    return pdf, png


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    print(f"Simulating {N_FRAMES} OFDM frames at each of {len(EBN0_DB)} Eb/N0 values ...")
    frames = simulate_frames()
    n_cfg = len(WAVELETS) * len(LEVELS) * len(core.THRESHOLD_RULES)
    print(f"Evaluating {n_cfg} wavelet configurations under AWGN and impulsive noise ...")
    ber = evaluate_ber(frames)
    ber.to_csv(OUT / "thresholding_ber_all_configurations.csv", index=False)

    table = fig12_table(ber)
    table.to_csv(OUT / "fig12_ber_db4_J3.csv", index=False)
    saved = []
    for noise in NOISES:
        fig = plot_fig12_panel(table, noise)
        saved += save_figure(fig, FIG12_FILES[noise])
        plt.close(fig)
    for noise in NOISES:
        fig = plot_sensitivity_grid(ber, noise)
        saved += save_figure(fig, GRID_FILES[noise])
        plt.close(fig)

    print(f"\nBER plotted in Fig. 12 ({FIG12_WAVELET}, J = {FIG12_LEVEL}, threshold 4 x MAD; "
          f"{frames[0].bits.size} bits per Eb/N0 value)\n")
    shown = table.set_index("EbN0_dB")
    shown.index = shown.index.map(lambda v: f"{v:g}")
    for noise, title in (("AWGN", "(a) AWGN"), ("Middleton", "(b) Middleton Class A impulsive noise")):
        part = shown[[f"{noise} - {c}" for c in CURVES]].rename(columns=lambda s: s.split(" - ")[1])
        print(title)
        print(part.rename_axis("Eb/N0 (dB)").apply(lambda col: col.map("{:.3e}".format)).to_string())
        print()
    print("Figures saved:", *[p.name for p in saved], sep="\n  ")
    print(f"\nFinished in {time.perf_counter() - t0:.0f} s. Results are in {OUT}")


if __name__ == "__main__":
    main()
