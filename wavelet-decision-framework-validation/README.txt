# Receiver-Level Validation of the Unified Decision Framework for Wavelet-Parameter Selection

Supplementary code for:

> P. Sheela and I. Dey, "Advances in Wavelet Denoising for Communication Signals: From Parameter Selection Toward Data-Driven Optimization," *IEEE Open Journal of the Communications Society*, to appear.

The code reproduces the receiver-level validation of the proposed decision framework:

- **Synthetic case study:** pilot-aided OFDM channel estimation.
- **Measured case study:** over-the-air IEEE 802.11n channels.
- **Significance tests:** the Wilcoxon signed-rank tests reported for both case studies.

---

## 1. Description

The framework chooses a wavelet-denoising configuration: transform family, mother wavelet, decomposition level and thresholding strategy. It judges every candidate by receiver-level performance. For each case study and noise regime, the code does the following:

1. Generates OFDM frames and splits them into a **validation** set (40 %) and a **held-out test** set (60 %).
2. Selects the wavelet configuration on the validation set, using receiver metrics only.
3. Evaluates the selected configuration on the test set against three baselines: raw least squares (LS), a tuned DFT estimator and a fixed-parameter wavelet.
4. Reports the bit error rate (BER), error vector magnitude (EVM), channel-estimation NMSE and complex SNR gain, together with Wilcoxon signed-rank significance tests.


There are two ways to run the validation:

- **Scripts.** Each case study is a Python script that runs end to end.
- **Notebook.** The Jupyter notebook `stage_by_stage_validation.ipynb` runs the same stages one at a time, so you can inspect the intermediate results.


## 2. Size

| File | Size |
|---|---|
| `README.txt` | 17 KB |
| `decision_framework_core.py` | 15 KB |
| `synthetic_ofdm_validation.py` | 9 KB |
| `measured_csi_validation.py` | 8 KB |
| `stage_by_stage_validation.ipynb` | 74 KB |
| `data/measured_80211n_csi_pool.npy` | 390 KB |
| `data/LICENSE_csi_classification.txt` | 2 KB |
| `requirements.txt`| 1 KB |
| **Total** | **about 520 KB** |

The `results/` folder created by a run adds about 1 KB.

## 3. Platform

The package is pure Python with no compiled code, so it runs on any operating system with a supported Python: Windows, macOS or Linux. 


## 4. Environment

| Component | Requirement | Versions verified (identical results) |
|---|---|---|
| Python | ≥ 3.9 | 3.9.25, 3.11.17, 3.12.3, 3.13.16 |
| NumPy | ≥ 1.26 | 1.26.4, 2.5.3, 2.5.4 |
| SciPy | ≥ 1.11 | 1.11.4, 1.18.1 |
| pandas | ≥ 2.0 | 2.0.3, 3.0.5, 3.0.6 |
| PyWavelets | ≥ 1.5 | 1.5.0, 1.10.0 |
| ipykernel (notebook only) | any recent | 6.31.0 (Python 3.9), 7.4.0 (Python 3.12) |

No GPU, compiler or internet connection is needed after installation. Running both case studies takes about 10 s on a standard PC, and running the notebook takes about 15 s.

Visual Studio Code is optional. If you use it, install the **Python** and **Jupyter** extensions from Microsoft. 

## 5. Major component description

| File | Purpose |
|---|---|
| `decision_framework_core.py` | Shared library. It provides: <br>• the OFDM frame model, QPSK, and the AWGN and bursty-impulsive noise models; <br>• the stratified validation/test split; <br>• the channel estimators: LS, DFT, wavelet shrinkage (DWT/SWT/WPT), decision-directed refinement, robust clipping and LMMSE; <br>• MMSE/ZF equalisation and the receiver metrics; <br>• the framework's configuration selection. |
| `synthetic_ofdm_validation.py` | Synthetic case study |
| `measured_csi_validation.py` | Measured-channel case study|
| `stage_by_stage_validation.ipynb` | Jupyter notebook. It imports the three program files and runs each stage separately, showing the intermediate results: <br>• data; <br>• validation/test split; <br>• candidate grid; <br>• the five best-ranked candidates and the delay-support tuning; <br>• test performance; <br>• the full Holm family. <br>It ends with a cell-by-cell comparison against the article. |
| `data/measured_80211n_csi_pool.npy` | 445 measured complex channel vectors × 56 subcarriers (IEEE 802.11n, Atheros) |
| `data/LICENSE_csi_classification.txt` | Source and licence of the measured data |


Each case-study script is organised into stage functions:

| Stage | Synthetic study | Measured study |
|---|---|---|
| 1. Data | `simulate_dataset()` | `load_pool()`, `build_frames()` |
| 2. Validation/test split | `split_frames()` | `split_frames()` |
| 3. Selection on the validation set | `select_on_validation()` | `select_on_validation()` |
| 4. Held-out test metrics | `evaluate_on_test()` | `evaluate_on_test()` |
| 5. Tables | `performance_table()`, `significance_table()` | `performance_table()`, `significance_table()` |

`main()` runs these stages in order. The notebook calls the same functions one at a time.

### 5.1 Receiver chain (both case studies)
1. **LS estimate.** Least-squares (LS) channel estimates are taken at the comb pilots (every 4th subcarrier). Circular linear interpolation fills in the other subcarriers.
2. **Denoising in the delay domain** (inverse FFT of the LS estimate):
   - **Raw LS:** no denoising.
   - **DFT tuned:** keeps the first *K* delay taps, with *K* tuned on validation.
   - **Fixed wavelet:** DWT, db4, J = 2, sqtwolog, soft thresholding, I and Q processed separately.
   - **Proposed framework:** the configuration the framework selects.
3. **Decision-directed refinement.** This applies to the DFT, fixed-wavelet and framework rows. There are two iterations. In each, tentative QPSK decisions act as additional pilots, the channel is re-estimated on all subcarriers and cut to the first *K* delay taps (N/4 if *K* is 0 or ≥ N). Refinement is skipped for frames whose estimated SNR is below 8 dB.
4. **MMSE equalisation**, with the noise power estimated from the LS delay taps beyond the support length.
5. **Metrics**, computed per frame and averaged over the test frames:
   - BER;
   - EVM (%);
   - channel NMSE = Σ|H − Ĥ|² / Σ|H|²;
   - complex SNR gain of the channel estimate relative to raw LS, 10·log10(err_LS / err_method). Raw LS = 0 dB by construction.

### 5.2 Framework configuration selection (validation set only)
- **Candidate set.** Every admissible combination of:
  - transform family {DWT, SWT, WPT};
  - mother wavelet;
  - decomposition level J;
  - threshold rule {sqtwolog, rigrsure};
  - shrinkage {soft, garrote};
  - threshold scale {0.5, 1.0};
  - complex-coefficient handling {I/Q separately, magnitude–phase}.
- **Decomposition level (synthetic study).** The band-aligned depth J_band = ⌈log₂((f_s/2)/B)⌉ = 2 and its neighbours, i.e. J ∈ {1, 2, 3}.
- **Ranking.** Each candidate is scored with a zero-forcing receiver and ranked by BER, then EVM, then NMSE.
- **Delay-support length.** The length for the winning configuration is then tuned by NMSE (0 = off).

| | Synthetic study | Measured study |
|---|---|---|
| Subcarriers | 128 (32 pilots / 96 data), CP 32 | 56 (14 pilots / 42 data), channel applied in the frequency domain |
| Channel | 12-tap exponential-PDP Rayleigh, renormalised per frame | 445 measured 802.11n channels, unit average power |
| Sampling rate / bandwidth | 30.72 MHz / 5 MHz | – |
| SNR | 0, 5, 10, 15, 20 dB | 0, 5, 10, 15, 20 dB |
| Noise regimes | AWGN; bursty-impulsive (Gaussian + sparse impulses, p = 0.02, +20 dB) | same |
| Frames | 80 per (regime, SNR); 160 validation / 240 test per regime | same |
| Candidate configurations | 512 (haar, db4, sym4, dmey) | 216 (haar, db4, sym4) |
| Selected configuration (both regimes) | WPT, haar, J = 2, sqtwolog, soft, scale 1.0, magnitude–phase, support 12 taps | SWT, haar, J = 3, sqtwolog, soft, scale 1.0, I/Q separately, support off |
| Seeds | data 7, split 10 | data 7 (AWGN), 11 (impulsive); split 10 |

### 5.3 Significance tests
All tests are paired, two-sided Wilcoxon signed-rank tests over the 240 test frames of a regime.
- **Synthetic study.** The test uses per-frame BER. The framework's wavelet estimate is compared with each of the other 8 estimators: raw LS, DFT, fixed wavelet, robust clipping and LMMSE, plus the decision-directed variants. The Holm correction is applied over these 8 comparisons. The comparisons with the fixed wavelet and with raw LS are reported.
- **Measured study.** The test uses per-frame channel NMSE of the AWGN test frames. The framework's wavelet estimate is compared with raw LS and with the fixed wavelet (estimates before decision-directed refinement).

## 6. Setup instructions

Install Python 3.9 or newer from <https://www.python.org>. On Windows, tick **"Add Python to PATH"** during installation.

Open a terminal in this folder and run:

```
python -m venv .venv
.venv\Scripts\activate             (Windows)
source .venv/bin/activate          (Linux / macOS)
pip install -r requirements.txt
pip install ipykernel              (only needed for the notebook)
```

On Windows, if `python` is not recognised, use `py` instead. On Linux and macOS, `python3` may be needed.

## 7. Run instructions

### 7.1 Terminal only

With the environment activated:

- **Windows:** `run_all.bat`, or double-click the file in File Explorer.
- **Linux / macOS:** `./run_all.sh`
- **Individually:**
  ```
  python synthetic_ofdm_validation.py
  python measured_csi_validation.py
  ```
- **Notebook:** `pip install notebook`, then `jupyter notebook stage_by_stage_validation.ipynb`.

### 7.2 Stage-by-stage notebook

`stage_by_stage_validation.ipynb` has two parts.

**Part A: synthetic study**

| Stage | Content |
|---|---|
| A1 | Frames |
| A2 | Split |
| A3 | Candidate grid and J_band |
| A4 | Validation selection, top-5 candidates and support tuning |
| A5 | Table 13 |
| A6 | Holm family and p-values |

**Part B: real-measured channel study**

| Stage | Content |
|---|---|
| B1 | Channel pool |
| B2 | Frames and split |
| B3 | Validation selection |
| B4 | Real measured channel|
| B5 | Wilcoxon p-values |


```

## 8. Data source

`data/measured_80211n_csi_pool.npy` was extracted from the IEEE 802.11n channel-state-information logs of the public repository [maksimio/csi_classification](https://github.com/maksimio/csi_classification), captured with an Atheros CSI tool. The MIT licence of that repository is reproduced in `data/LICENSE_csi_classification.txt`.

- **Source files:** `csi/homelocation/five place/test/bathroom4.dat` and `kitchen4.dat`, and `csi/homelocation/three place/test/outer10.dat`.
- **Processing:** antenna pair rx0/tx0 was kept, along with every 10th packet, giving 445 complex vectors × 56 subcarriers.

## 9. Citation

```bibtex
@article{sheela_wavelet_denoising,
  author  = {Sheela, Priyalakshmi and Dey, Indrakshi},
  title   = {Advances in Wavelet Denoising for Communication Signals: From Parameter
             Selection Toward Data-Driven Optimization},
  journal = {IEEE Open Journal of the Communications Society},
  note    = {to appear}
}
```

## 10. Contact information

Priyalakshmi Sheela, Walton Institute, South East Technological University, Waterford, Ireland
E-mail: priyalakshmi.sheela@setu.ie
