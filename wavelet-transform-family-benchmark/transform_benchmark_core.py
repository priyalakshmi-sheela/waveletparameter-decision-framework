"""
transform_benchmark_core.py - OFDM link, Middleton Class-A impulsive noise,
DWT / SWT / WPT impulse-suppression pipelines and the metrics used by
transform_family_benchmark.py.

Contents
  * OFDM transmitter / receiver: 64-point FFT, 16-sample cyclic prefix,
    48 active QPSK subcarriers (IEEE 802.11a data-subcarrier layout)
  * complex Middleton Class-A impulsive noise
  * wavelet impulse suppression: the in-phase and quadrature components are
    processed separately; in every detail subband the coefficients whose
    magnitude reaches the threshold are taken as the impulse estimate
    (hard / soft / garrote shaped), the impulse estimate is reconstructed and
    subtracted from the received waveform
  * waveform metrics (complex SNR gain, NMSE), receiver metrics (BER, RMS EVM)
  * FLOP proxy per frame

Conventions: waveform arrays are (frames, samples) or a single frame
(samples,); every wavelet operation runs along the last axis, so a batch of
frames gives exactly the same numbers as processing the frames one by one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pywt

NFFT, NCP = 64, 16
ACTIVE_BINS = np.r_[np.arange(-26, -21), np.arange(-20, -7), np.arange(-6, 0),
                    np.arange(1, 7), np.arange(8, 21), np.arange(22, 27)]      # 48 data subcarriers
_ACTIVE_COLS = ACTIVE_BINS + NFFT // 2          # positions of the active bins in the fftshift-ed spectrum


# ------------------------------------------------------------------ OFDM link
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits).reshape(-1, 2)
    return ((2 * b[:, 0] - 1) + 1j * (2 * b[:, 1] - 1)) / np.sqrt(2)


def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """Hard decisions along the last axis (bit pairs interleaved as in qpsk_mod)."""
    s = np.asarray(symbols)
    bits = np.stack([(np.real(s) > 0), (np.imag(s) > 0)], axis=-1).astype(int)
    return bits.reshape(*s.shape[:-1], -1)


def ofdm_modulate(bits: np.ndarray) -> np.ndarray:
    """One frame: QPSK on the active subcarriers, IFFT, cyclic prefix, serialised."""
    sym = qpsk_mod(bits).reshape(-1, len(ACTIVE_BINS))
    out = []
    for row in sym:
        x_shift = np.zeros(NFFT, dtype=complex)
        x_shift[_ACTIVE_COLS] = row
        x = np.fft.ifft(np.fft.ifftshift(x_shift))
        out.append(np.r_[x[-NCP:], x])
    return np.concatenate(out)


def ofdm_demodulate(waveform: np.ndarray) -> np.ndarray:
    """Active-subcarrier symbols of a frame (or of every row of a batch),
    in the order produced by qpsk_mod."""
    w = np.asarray(waveform)
    blocks = w.reshape(*w.shape[:-1], -1, NFFT + NCP)[..., NCP:]
    spec = np.fft.fftshift(np.fft.fft(blocks, axis=-1), axes=-1)[..., _ACTIVE_COLS]
    return spec.reshape(*w.shape[:-1], -1)


def middleton_class_a(rng: np.random.Generator, n: int, sigma2_total: float, A: float, T: float,
                      mixture_terms: int = 30) -> np.ndarray:
    """Complex Middleton Class-A noise; A = impulsive index, T = Gaussian-to-impulsive power ratio."""
    beta = np.array([math.exp(-A) * (A ** m) / math.factorial(m) for m in range(mixture_terms)])
    beta /= beta.sum()
    m_idx = rng.choice(np.arange(mixture_terms), size=n, p=beta)
    sigma2_m = ((m_idx / A) + T) / (1 + T) * sigma2_total
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(sigma2_m / 2)


# -------------------------------------------------------------------- metrics
def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """Complex SNR (dB) of `test` with respect to `clean` (Table 9, entries 1, 2 and 13)."""
    return float(10.0 * np.log10(np.sum(np.abs(clean) ** 2) / (np.sum(np.abs(clean - test) ** 2) + 1e-15)))


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Waveform-domain NMSE. The transmitted frame has unit average power, so NMSE = MSE."""
    return float(np.sum(np.abs(reference - estimate) ** 2) / (np.sum(np.abs(reference) ** 2) + 1e-15))


def evm_rms_percent(reference: np.ndarray, received: np.ndarray) -> float:
    """RMS EVM (%) after removing one least-squares complex gain, which absorbs the
    fixed scaling between the unit-power waveform and the constellation."""
    alpha = np.vdot(received, reference) / (np.vdot(received, received) + 1e-15)
    est = alpha * received
    return float(100.0 * np.sqrt(np.sum(np.abs(reference - est) ** 2) / (np.sum(np.abs(reference) ** 2) + 1e-15)))


def ci95(values) -> float:
    """Half-width of the 95 % confidence interval of the mean (normal approximation)."""
    v = np.asarray(values, dtype=float)
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


# ------------------------------------------------------ wavelet configuration
@dataclass(frozen=True)
class TransformConfig:
    transform: str            # "DWT" | "SWT" | "WPT"
    wavelet: str              # "db4" | "sym4" | "coif4"
    level: int                # decomposition depth J
    threshold_function: str   # "scaled_mad" | "universal" | "bayes"
    threshold_rule: str       # "hard" | "soft" | "garrote"
    threshold_scale: float = 2.5   # multiplier of the MAD noise estimate (scaled_mad only)

    def label(self) -> str:
        fn = (f"{self.threshold_function} x{self.threshold_scale:g}" if self.threshold_function == "scaled_mad"
              else self.threshold_function)
        return f"{self.transform}/{self.wavelet}/J={self.level}/{fn}/{self.threshold_rule}"


# ------------------------------------------------ thresholding (per subband)
def _square(v: np.ndarray) -> np.ndarray:
    """Element-wise v**2 computed with Python's scalar power (C pow), as in the
    reference implementation; NumPy's array square can differ in the last bit."""
    v = np.asarray(v, dtype=float)
    return np.array([float(x) ** 2 for x in v.ravel()]).reshape(v.shape)


def subband_threshold(c: np.ndarray, function: str, scale: float) -> np.ndarray:
    """Threshold of one subband (one value per frame), from the MAD noise
    estimate sigma = median(|c|) / 0.6745 of that subband."""
    sigma = np.median(np.abs(c), axis=-1) / 0.6745 + 1e-12
    if function == "scaled_mad":
        return scale * sigma
    if function == "universal":
        return sigma * np.sqrt(2.0 * np.log(max(c.shape[-1], 2)))
    if function == "bayes":                                   # BayesShrink
        sigma2 = _square(sigma)
        sigma_x = np.sqrt(np.maximum(np.mean(np.abs(c) ** 2, axis=-1) - sigma2, 0.0))
        flat = sigma_x <= 1e-15                               # no signal energy: remove nothing
        return np.where(flat, np.max(np.abs(c), axis=-1) + 1.0, sigma2 / np.where(flat, 1.0, sigma_x))
    raise ValueError(f"unsupported threshold function {function!r}")


def impulse_part(c: np.ndarray, thr: np.ndarray, rule: str) -> np.ndarray:
    """Coefficients at or above the threshold, shaped by the threshold rule
    (the impulse estimate); all other coefficients are set to zero."""
    t = np.asarray(thr)[..., None]
    mag = np.abs(c)
    keep = mag >= t
    if rule == "hard":
        return np.where(keep, c, 0.0)
    if rule == "soft":
        return np.where(keep, c * (np.maximum(mag - t, 0.0) / (mag + 1e-15)), 0.0)
    if rule == "garrote":
        return np.where(keep, c * np.maximum(1.0 - _square(t) / (mag ** 2 + 1e-15), 0.0), 0.0)
    raise ValueError(f"unsupported threshold rule {rule!r}")


# ----------------------------------------------------- impulse suppression
def _impulse_estimate(x: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    """Reconstructed impulse estimate of a real-valued waveform (approximation subband excluded)."""
    w, lvl = cfg.wavelet, cfg.level

    def part(c):
        return impulse_part(c, subband_threshold(c, cfg.threshold_function, cfg.threshold_scale),
                            cfg.threshold_rule)

    if cfg.transform == "DWT":
        c = pywt.wavedec(x, w, mode="periodization", level=lvl, axis=-1)
        est = pywt.waverec([np.zeros_like(c[0])] + [part(d) for d in c[1:]], w, mode="periodization", axis=-1)
    elif cfg.transform == "SWT":
        c = pywt.swt(x, w, level=lvl, start_level=0, trim_approx=False, axis=-1)
        est = pywt.iswt([(np.zeros_like(a), part(d)) for a, d in c], w, axis=-1)
    elif cfg.transform == "WPT":
        leaves = pywt.WaveletPacket(x, w, mode="periodization", maxlevel=lvl, axis=-1).get_level(lvl, order="freq")
        tree = pywt.WaveletPacket(None, w, mode="periodization", maxlevel=lvl, axis=-1)
        for node in leaves:
            tree[node.path] = part(node.data)
        est = tree.reconstruct(update=False)
    else:
        raise ValueError(f"unsupported transform {cfg.transform!r}")
    return est[..., :x.shape[-1]]


def suppress_impulses(y: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    """Denoised complex waveform: I and Q each minus their reconstructed impulse estimate."""
    yr, yi = np.real(y), np.imag(y)
    return (yr - _impulse_estimate(yr, cfg)) + 1j * (yi - _impulse_estimate(yi, cfg))


def level_supported(cfg: TransformConfig, n: int) -> bool:
    if cfg.transform == "SWT":
        return n % (2 ** cfg.level) == 0
    return cfg.level <= pywt.dwt_max_level(n, pywt.Wavelet(cfg.wavelet).dec_len)


# ----------------------------------------------------------------- FLOP proxy
def flop_proxy(cfg: TransformConfig, n: int) -> float:
    """Deterministic multiply-accumulate proxy per frame of n complex samples:
    DWT: 4 Lf sum_{j=1..J} n / 2^(j-1);  SWT and WPT: 8 Lf n J  (Lf = filter length).
    These equal the article's 2 Lf N sum_j 2^-j and 2 Lf N J with N = 4n."""
    lf = pywt.Wavelet(cfg.wavelet).dec_len
    if cfg.transform == "DWT":
        return 4.0 * lf * sum(n / (2 ** (j - 1)) for j in range(1, cfg.level + 1))
    if cfg.transform in ("SWT", "WPT"):
        return 8.0 * lf * n * cfg.level
    raise ValueError(cfg.transform)
