"""
decision_framework_core.py - receiver chain and wavelet-parameter selection
shared by synthetic_ofdm_validation.py and measured_csi_validation.py.

Contents
  * OFDM frame container, QPSK mapping, AWGN / bursty-impulsive noise
  * stratified validation/test split
  * channel estimators: pilot LS, DFT (delay-tap truncation), wavelet
    shrinkage (DWT / SWT / WPT), decision-directed refinement, and two
    further baselines (robust clipping, covariance LMMSE)
  * MMSE / ZF equalisation and receiver-level metrics (BER, EVM, NMSE,
    complex SNR gain)
  * selection of the wavelet configuration {transform, mother wavelet,
    level, threshold rule, shrinkage} on the validation split

Conventions: arrays are (frames, subcarriers); all frames of a split are
processed together as one 2-D batch; "delay domain" = IFFT along the
subcarrier axis. Summation orders are fixed so that results are
bit-reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pywt

SQRT2 = np.sqrt(2.0)
PILOT_SYMBOL = (1.0 + 1.0j) / SQRT2


# ---------------------------------------------------------------- data types
@dataclass(frozen=True)
class WaveletConfig:
    """One point of the {F, psi, J, T} search space (+ delay-support gate)."""
    transform: str            # "DWT" | "SWT" | "WPT"
    wavelet: str              # e.g. "haar", "db4", "sym4", "dmey"
    level: int                # decomposition depth J
    threshold_rule: str       # "sqtwolog" | "rigrsure"
    shrinkage: str            # "soft" | "garrote" | "hard"
    threshold_scale: float = 1.0
    complex_mode: str = "component"   # "component" (I/Q separately) | "magnitude_phase"
    support_taps: int = 0             # keep first K delay taps after shrinkage; 0 = off

    def label(self) -> str:
        return (f"{self.transform}/{self.wavelet}/J={self.level}/{self.threshold_rule}/"
                f"{self.shrinkage}/scale={self.threshold_scale}/{self.complex_mode}/"
                f"support={self.support_taps}")


@dataclass
class FrameBatch:
    """A set of OFDM frames stored as arrays, with per-frame caches."""
    frame_id: np.ndarray      # (F,)
    snr_db: np.ndarray        # (F,)
    x: np.ndarray             # (F, N) transmitted frequency-domain symbols
    y: np.ndarray             # (F, N) received frequency-domain symbols
    h: np.ndarray             # (F, N) true channel frequency response
    bits: np.ndarray          # (F, 2*Nd) transmitted data bits
    pilot_idx: np.ndarray
    data_idx: np.ndarray

    def __post_init__(self):
        self.h_ls = ls_estimate(self)                                   # interpolated LS
        self.sig_pow = np.sum(np.abs(self.h) ** 2, axis=1) + 1e-12       # |H|^2
        self.ls_err = np.sum(np.abs(self.h - self.h_ls) ** 2, axis=1) + 1e-12

    @property
    def n_sc(self) -> int:
        return self.x.shape[1]

    def subset(self, rows: np.ndarray) -> "FrameBatch":
        rows = np.asarray(rows, dtype=int)
        return FrameBatch(self.frame_id[rows], self.snr_db[rows], self.x[rows], self.y[rows],
                          self.h[rows], self.bits[rows], self.pilot_idx, self.data_idx)


# ------------------------------------------------------------ helpers
def _rowsum(a: np.ndarray) -> np.ndarray:
    """Row sums with a fixed 1-D summation order (NumPy may reorder a 2-D
    reduction over fancy-indexed columns), for bit-reproducible results."""
    return np.array([np.sum(r) for r in a])


# ------------------------------------------------------------ modulation/grid
def pilot_data_indices(n_sc: int, spacing: int):
    pilot = np.arange(0, n_sc, spacing)
    return pilot, np.setdiff1d(np.arange(n_sc), pilot)


def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits).astype(int).reshape(*np.shape(bits)[:-1], -1, 2)
    return ((1.0 - 2.0 * b[..., 0]) + 1j * (1.0 - 2.0 * b[..., 1])) / SQRT2


def qpsk_demod(sym: np.ndarray) -> np.ndarray:
    out = np.empty(sym.shape[:-1] + (2 * sym.shape[-1],), dtype=int)
    out[..., 0::2] = np.real(sym) < 0
    out[..., 1::2] = np.imag(sym) < 0
    return out


# ------------------------------------------------------------------ noise
def _cgauss(rng, shape, power):
    return np.sqrt(power / 2.0) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def add_noise(rng, signal: np.ndarray, snr_db: float, model: str) -> np.ndarray:
    """AWGN, or AWGN + sparse impulses (p = 0.02, impulses 20 dB above background).
    Total noise power always equals signal power / SNR."""
    p_sig = np.mean(np.abs(signal) ** 2)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    if model == "awgn":
        return signal + _cgauss(rng, signal.shape, p_noise)
    if model == "bursty_impulsive":
        prob, ratio = 0.02, 10.0 ** (20.0 / 10.0)
        p_bg = p_noise / (1.0 + ratio * prob)
        background = _cgauss(rng, signal.shape, p_bg)
        mask = rng.random(signal.shape) < prob
        impulses = _cgauss(rng, signal.shape, p_bg * ratio) * mask
        return signal + background + impulses
    raise ValueError(f"unknown noise model {model!r}")


# ---------------------------------------------------- split (stratified 40/60)
def split_validation_test(snr_db: np.ndarray, regime: np.ndarray, val_fraction=0.4, seed=10):
    """40/60 validation/test split, stratified by (noise regime, SNR)."""
    rng = np.random.default_rng(seed)
    val = []
    for key in sorted(set(zip(regime.tolist(), snr_db.tolist()))):
        idx = np.flatnonzero((regime == key[0]) & (snr_db == key[1]))
        rng.shuffle(idx)
        val.extend(idx[: int(val_fraction * len(idx))].tolist())
    is_val = np.zeros(len(snr_db), dtype=bool)
    is_val[val] = True
    return np.flatnonzero(is_val), np.flatnonzero(~is_val)


# ------------------------------------------------------------ estimators
def ls_estimate(fb: FrameBatch) -> np.ndarray:
    """Pilot LS + circular linear interpolation over all subcarriers."""
    n, p = fb.n_sc, fb.pilot_idx
    hp = fb.y[:, p] / (fb.x[:, p] + 1e-12)
    xp = np.concatenate([p - n, p, p + n])
    grid = np.arange(n)
    out = np.empty(fb.y.shape, dtype=complex)
    for i in range(hp.shape[0]):
        v = np.concatenate([hp[i], hp[i], hp[i]])
        out[i] = np.interp(grid, xp, np.real(v)) + 1j * np.interp(grid, xp, np.imag(v))
    return out


def dft_estimate(h_ls: np.ndarray, keep: int) -> np.ndarray:
    """Delay-domain truncation to the first `keep` taps."""
    keep = int(max(1, min(keep, h_ls.shape[1])))
    hd = np.fft.ifft(h_ls, axis=1)
    out = np.zeros_like(hd)
    out[:, :keep] = hd[:, :keep]
    return np.fft.fft(out, axis=1)


def noise_variance(fb: FrameBatch, support: int | None) -> np.ndarray:
    """Frequency-domain N0 from the out-of-support LS delay taps (used by MMSE)."""
    hd = np.fft.ifft(fb.h_ls, axis=1)
    n = fb.n_sc
    k = int(support) if (support is not None and 0 < int(support) < n) else max(1, n // 4)
    return np.mean(np.abs(hd[:, k:]) ** 2, axis=1) * n


def decision_directed(fb: FrameBatch, h_init: np.ndarray, support: int, iters=2, gate_db=8.0):
    """Tentative QPSK decisions reused as pilots, re-estimate on all subcarriers,
    keep the first K delay taps. Skipped for frames whose estimated SNR < gate."""
    n, d, p = fb.n_sc, fb.data_idx, fb.pilot_idx
    k = int(support)
    if k <= 0 or k >= n:
        k = max(1, n // 4)
    n0 = noise_variance(fb, k)
    p_y = _rowsum(np.abs(fb.y[:, d]) ** 2) / len(d)
    active = 10.0 * np.log10(np.maximum(p_y - n0, 1e-12) / (n0 + 1e-12)) >= gate_db
    h = h_init
    for _ in range(int(iters)):
        x_hat = fb.x.astype(complex, copy=True)
        x_hat[:, d] = qpsk_mod(qpsk_demod(fb.y[:, d] / (h[:, d] + 1e-12)))
        x_hat[:, p] = fb.x[:, p]
        hd = np.fft.ifft(fb.y / (x_hat + 1e-12), axis=1)
        hd[:, k:] = 0.0
        h = np.where(active[:, None], np.fft.fft(hd, axis=1), h)
    return h


def clipping_estimate(h_ls: np.ndarray, factor: float) -> np.ndarray:
    """Robust delay-domain median/MAD clipping (additional baseline in the
    synthetic study's multiple-comparison family)."""
    hd = np.fft.ifft(h_ls, axis=1)
    mag = np.abs(hd)
    med = np.median(mag, axis=1, keepdims=True)
    thr = med + factor * (np.median(np.abs(mag - med), axis=1, keepdims=True) / 0.6745)
    out = np.fft.fft(hd * np.minimum(1.0, thr / (mag + 1e-12)), axis=1)
    return np.where(thr > 1e-12, out, h_ls)


def lmmse_estimate(fb_val: FrameBatch, h_ls: np.ndarray) -> np.ndarray:
    """Covariance LMMSE with the prior fitted on the validation split (additional
    baseline in the synthetic study's multiple-comparison family). Solved frame
    by frame for bit-reproducibility."""
    n = fb_val.n_sc
    r = (fb_val.h.conj().T @ fb_val.h) / max(fb_val.h.shape[0], 1) + 1e-8 * np.eye(n)
    sigma2 = float(np.mean(np.mean(np.abs(fb_val.h_ls - fb_val.h) ** 2, axis=1)))
    a = r + max(sigma2, 1e-12) * np.eye(n)
    return np.array([r @ np.linalg.solve(a, row) for row in h_ls])


# ------------------------------------------------------- wavelet shrinkage
def _mad_sigma(c: np.ndarray) -> np.ndarray:
    med = np.median(c, axis=1, keepdims=True)
    return np.median(np.abs(c - med), axis=1) / 0.6745


def _threshold(details: np.ndarray, rule: str) -> np.ndarray:
    """Per-row threshold from pooled detail coefficients (real, or magnitudes)."""
    n = details.shape[1]
    sigma = _mad_sigma(details)
    ok = sigma > 1e-12
    if rule == "sqtwolog":
        thr = sigma * np.sqrt(2.0 * np.log(n))
    elif rule == "rigrsure":
        s = np.where(ok, sigma, 1.0)[:, None]
        x = np.sort(np.abs(details / s), axis=1) ** 2
        k = np.arange(1, n + 1)
        risk = (n - 2 * k + np.cumsum(x, axis=1) + (n - k) * x) / n
        thr = sigma * np.sqrt(x[np.arange(len(x)), np.argmin(risk, axis=1)])
    else:
        raise ValueError(f"unsupported threshold rule {rule!r}")
    return np.where(ok, thr, 0.0)


def _shrink(c: np.ndarray, thr: np.ndarray, mode: str, complex_mag: bool) -> np.ndarray:
    t = thr[:, None]
    if complex_mag:                                   # phase-preserving
        mag = np.abs(c)
        if mode == "soft":
            out = np.maximum(1.0 - t / (mag + 1e-12), 0.0) * c
        elif mode == "garrote":
            out = np.maximum(1.0 - (t ** 2) / (mag ** 2 + 1e-12), 0.0) * c
        else:
            out = np.where(mag >= t, c, 0.0 + 0.0j)
    else:                                             # identical to pywt.threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            out = pywt.threshold(c, t, mode=mode)
    return np.where(t > 0, out, c)


def _denoise(x: np.ndarray, cfg: WaveletConfig, complex_mag: bool) -> np.ndarray:
    """One DWT/SWT/WPT shrinkage pass over every row of x."""
    w, lvl, n = cfg.wavelet, cfg.level, x.shape[1]
    pool = (lambda arrs: np.concatenate([np.abs(a) for a in arrs], axis=1)) if complex_mag \
        else (lambda arrs: np.concatenate(arrs, axis=1))

    def thr_of(arrs):
        return float(cfg.threshold_scale) * _threshold(pool(arrs), cfg.threshold_rule)

    if cfg.transform == "DWT":
        c = pywt.wavedec(x, w, level=lvl, mode="periodization", axis=-1)
        t = thr_of(c[1:])
        rec = pywt.waverec([c[0]] + [_shrink(d, t, cfg.shrinkage, complex_mag) for d in c[1:]],
                           w, mode="periodization", axis=-1)
    elif cfg.transform == "SWT":
        c = pywt.swt(x, w, level=lvl, trim_approx=False, norm=False, axis=-1)
        t = thr_of([cd for _, cd in c])
        rec = pywt.iswt([(ca, _shrink(cd, t, cfg.shrinkage, complex_mag)) for ca, cd in c], w, axis=-1)
    elif cfg.transform == "WPT":
        wp = pywt.WaveletPacket(data=x, wavelet=w, mode="periodization", maxlevel=lvl, axis=-1)
        nodes = wp.get_level(lvl, order="freq")
        t = thr_of([nd.data for nd in nodes])
        for nd in nodes:
            nd.data = _shrink(nd.data, t, cfg.shrinkage, complex_mag)
        rec = wp.reconstruct(update=True)
    else:
        raise ValueError(cfg.transform)
    return rec[:, :n]


def wavelet_estimate(h_ls: np.ndarray, cfg: WaveletConfig) -> np.ndarray:
    """Wavelet shrinkage of the LS channel in the delay domain (+ optional support gate)."""
    hd = np.fft.ifft(h_ls, axis=1)
    if cfg.complex_mode == "magnitude_phase":
        hw = _denoise(hd, cfg, complex_mag=True)
    else:
        hw = _denoise(np.real(hd), cfg, False) + 1j * _denoise(np.imag(hd), cfg, False)
    if 0 < cfg.support_taps < hw.shape[1]:
        hw = np.asarray(hw, dtype=complex).copy()
        hw[:, cfg.support_taps:] = 0.0
    return np.fft.fft(hw, axis=1)


def candidate_grid(n_sc, transforms, wavelets, levels, rules, shrinkages, scales, modes):
    """All admissible (wavelet, level) pairs x thresholding options."""
    out = []
    for tr, w in product(transforms, wavelets):
        wave = pywt.Wavelet(w)
        jmax = pywt.swt_max_level(n_sc) if tr == "SWT" else pywt.dwt_max_level(n_sc, wave.dec_len)
        for lvl, rule, shr, sc, mode in product([l for l in levels if 1 <= l <= jmax],
                                                rules, shrinkages, scales, modes):
            out.append(WaveletConfig(tr, w, int(lvl), rule, shr, float(sc), mode))
    return out


# ---------------------------------------------------------------- metrics
def receiver_metrics(fb: FrameBatch, h_est: np.ndarray, equalizer="zf", noise_support=None):
    """Per-frame BER, EVM (%), channel-estimation NMSE and complex SNR gain of the
    channel estimate relative to raw LS (dB)."""
    d = fb.data_idx
    y, h = fb.y[:, d], h_est[:, d]
    if equalizer == "mmse":
        lam = np.maximum(noise_variance(fb, noise_support), 0.0)[:, None]
        s = np.conj(h) * y / (np.abs(h) ** 2 + lam + 1e-12)
    else:
        s = y / (h + 1e-12)
    tx = fb.x[:, d]
    err_out = np.sum(np.abs(fb.h - h_est) ** 2, axis=1) + 1e-12
    return {
        "ber": np.mean(qpsk_demod(s) != fb.bits, axis=1),
        "evm_percent": 100.0 * np.sqrt(_rowsum(np.abs(tx - s) ** 2)
                                       / (_rowsum(np.abs(tx) ** 2) + 1e-12)),
        "nmse": err_out / fb.sig_pow,
        "snr_gain_db": 10.0 * np.log10(fb.sig_pow / err_out) - 10.0 * np.log10(fb.sig_pow / fb.ls_err),
    }


def plain_nmse(fb: FrameBatch, h_est: np.ndarray, eps_den=1e-12) -> np.ndarray:
    """Channel NMSE used for tuning the delay-support and DFT lengths."""
    return np.sum(np.abs(fb.h - h_est) ** 2, axis=1) / (np.sum(np.abs(fb.h) ** 2, axis=1) + eps_den)


def lexicographic_best(rows, keys=("ber", "evm_percent", "nmse")):
    """Index of the best row by (BER, EVM, NMSE); ties -> earliest candidate."""
    return min(range(len(rows)), key=lambda i: tuple(rows[i][k] for k in keys) + (i,))


def select_framework_config(fb_val: FrameBatch, grid, support_candidates):
    """Select {F1, W1, J1, T1} on the validation split: every candidate in `grid`
    is scored with a ZF receiver and ranked by BER, then EVM, then NMSE; the
    delay-support length of the winner is then tuned by NMSE."""
    scores = []
    for cfg in grid:
        m = receiver_metrics(fb_val, wavelet_estimate(fb_val.h_ls, cfg), "zf")
        scores.append({k: float(np.mean(v)) for k, v in m.items()})
    best = grid[lexicographic_best(scores)]
    support_nmse = [(float(np.mean(plain_nmse(fb_val, wavelet_estimate(fb_val.h_ls, replace(best, support_taps=k))))), k)
                    for k in support_candidates]
    return replace(best, support_taps=min(support_nmse)[1]), scores
