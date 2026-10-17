"""
thresholding_core.py - OFDM link, AWGN and Middleton Class-A noise, and the
wavelet impulse-suppression receiver used by thresholding_rule_sensitivity.py
(Fig. 12 and Supplementary Figs. S1 and S2 of the article).

Contents
  * OFDM transmitter / receiver: 64-point FFT, 16-sample cyclic prefix,
    48 active QPSK subcarriers (IEEE 802.11a data-subcarrier layout)
  * complex AWGN and complex Middleton Class-A impulsive noise
  * wavelet impulse suppression: the in-phase and quadrature components are
    processed separately; in every detail level the coefficients whose
    magnitude reaches the threshold (4 x MAD noise estimate of that level) are
    taken as the impulse estimate, shaped by the hard, soft or garrote rule;
    the impulse estimate is reconstructed and subtracted from the received waveform

Conventions: waveform arrays are (frames, samples) or a single frame
(samples,); every wavelet and FFT operation runs along the last axis, so a
batch of frames gives exactly the same numbers as processing the frames one by one.
"""
from __future__ import annotations

import math

import numpy as np
import pywt

NFFT, NCP = 64, 16
ACTIVE_BINS = np.r_[np.arange(-26, -21), np.arange(-20, -7), np.arange(-6, 0),
                    np.arange(1, 7), np.arange(8, 21), np.arange(22, 27)]      # 48 data subcarriers
_ACTIVE_COLS = ACTIVE_BINS + NFFT // 2          # positions of the active bins in the fftshift-ed spectrum
THRESHOLD_RULES = ("hard", "soft", "garrote")


# ------------------------------------------------------------------ OFDM link
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """Unit-energy QPSK; bit pairs (b0, b1) map to ((2 b0 - 1) + j (2 b1 - 1)) / sqrt(2)."""
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
    in the order produced by qpsk_mod: cyclic prefix removed, FFT, active bins."""
    w = np.asarray(waveform)
    blocks = w.reshape(*w.shape[:-1], -1, NFFT + NCP)[..., NCP:]
    spec = np.fft.fftshift(np.fft.fft(blocks, axis=-1), axes=-1)[..., _ACTIVE_COLS]
    return spec.reshape(*w.shape[:-1], -1)


# ---------------------------------------------------------------------- noise
def awgn(rng: np.random.Generator, n: int, sigma2_total: float) -> np.ndarray:
    """Circular complex white Gaussian noise of total power sigma2_total per sample."""
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(sigma2_total / 2)


def middleton_class_a(rng: np.random.Generator, n: int, sigma2_total: float, A: float, T: float,
                      mixture_terms: int = 30) -> np.ndarray:
    """Complex Middleton Class-A noise of total power sigma2_total per sample.
    A = impulsive index, T = Gaussian-to-impulsive power ratio; the Poisson
    mixture is truncated to `mixture_terms` terms."""
    beta = np.array([math.exp(-A) * (A ** m) / math.factorial(m) for m in range(mixture_terms)], dtype=float)
    beta /= beta.sum()
    m_idx = rng.choice(np.arange(mixture_terms), size=n, p=beta)
    sigma2_m = ((m_idx / A) + T) / (1 + T) * sigma2_total
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(sigma2_m / 2)


# ------------------------------------------------- wavelet impulse suppression
def mad_threshold(c: np.ndarray, scale: float) -> np.ndarray:
    """scale x sigma, with sigma = median(|c|) / 0.6745 of one detail level (one value per frame)."""
    sigma = np.median(np.abs(c), axis=-1, keepdims=True) / 0.6745 + 1e-12
    return scale * sigma


def threshold_coefficients(c: np.ndarray, thr: np.ndarray, rule: str) -> np.ndarray:
    """Coefficients kept as the impulse estimate (|c| >= thr), shaped by the rule:
    hard c; soft sgn(c)(|c| - thr); garrote c (1 - thr^2 / c^2)."""
    mag = np.abs(c)
    if rule == "hard":
        return np.where(mag >= thr, c, 0.0)
    if rule == "soft":
        return np.sign(c) * np.maximum(mag - thr, 0.0)
    if rule == "garrote":
        # thr^2 with the scalar power of the reference implementation (one value per frame)
        thr2 = np.array([t ** 2 for t in np.asarray(thr).ravel()]).reshape(np.shape(thr))
        return np.where(mag >= thr, c * (1.0 - thr2 / (mag ** 2 + 1e-12)), 0.0)
    raise ValueError(f"unknown threshold rule: {rule}")


def suppress_impulses(y: np.ndarray, wavelet: str, level: int, rule: str, threshold_scale: float = 4.0,
                      mode: str = "periodization") -> np.ndarray:
    """Wavelet impulse suppression of a received waveform (frame or batch of frames).
    For the in-phase and the quadrature component separately: DWT to `level`,
    impulse estimate from the thresholded detail coefficients (approximation set
    to zero), inverse DWT, and subtraction from the component."""
    y = np.asarray(y)
    out = np.empty_like(y)
    for part, x in enumerate((np.real(y), np.imag(y))):
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=mode, axis=-1)
        impulse = [np.zeros_like(coeffs[0])] + [threshold_coefficients(c, mad_threshold(c, threshold_scale), rule)
                                                 for c in coeffs[1:]]
        cleaned = x - pywt.waverec(impulse, wavelet=wavelet, mode=mode, axis=-1)[..., : x.shape[-1]]
        if part == 0:
            out.real = cleaned
        else:
            out.imag = cleaned
    return out


def level_supported(wavelet: str, level: int, n_samples: int) -> bool:
    """True if `level` does not exceed the maximum useful DWT level for the frame length."""
    return level <= pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
