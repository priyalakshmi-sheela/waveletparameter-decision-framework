# waveletparameter-decision-framework
Python implementations and reproducibility resources accompanying the IEEE OJ-COMS manuscript: Advances in Wavelet Denoising for Communication Signals: From Parameter Selection Toward Data-Driven Optimization


## Repository contents

| Folder | Description |
|---|---|
| [`Thresholding-rule-sensitivity/`] | Compares hard, soft, and garrote thresholding in an OFDM/QPSK receiver under AWGN and Middleton Class-A impulsive noise, including the sensitivity analysis for wavelet choice and decomposition level. |
| [`wavelet-decision-framework-validation/`]| Validates a wavelet-parameter selection framework through synthetic OFDM and measured IEEE 802.11n channel case studies, using receiver-level performance metrics and statistical tests. |
| [`wavelet-transform-family-benchmark/`]| Benchmarks discrete, stationary, and packet wavelet transforms (DWT, SWT, WPT) for impulsive-noise suppression, including performance, computational cost, and significance tests. |

## Getting started

Each folder is self-contained, with its own `README.txt`, `requirements.txt`, Python scripts, and Jupyter notebook. Refer to the [thresholding study](Thresholding-rule-sensitivity/README.txt), [decision-framework validation](wavelet-decision-framework-validation/README.txt), or [transform-family benchmark](wavelet-transform-family-benchmark/README.txt) for setup instructions, execution steps, and output descriptions.

**Requirements:** Python 3.9 or later. Install dependencies separately within the relevant folder using `pip install -r requirements.txt`.

## Citation

If you use this code in your research, please cite the accompanying article listed below:

P. Sheela and I. Dey, "Advances in Wavelet Denoising for Communication Signals: From Parameter Selection Toward Data-Driven Optimization," IEEE Open Journal of the Communications Society, early access, 2026, doi: 10.1109/OJCOMS.2026.3743854.
