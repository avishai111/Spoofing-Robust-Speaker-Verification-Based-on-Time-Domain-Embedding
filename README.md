<div align="center">

# Spoofing-Robust Speaker Verification Based on Time-Domain Embedding

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Research code for spoofing-robust automatic speaker verification using time-domain embeddings, gender-aware countermeasures, ECAPA-TDNN ASV scoring, and DCF/EER evaluation.

</div>

## Overview

This repository is a notebook-driven research workspace for spoofing-robust speaker verification on the ASVspoof2019 Logical Access database. It studies time-domain embeddings for countermeasure (CM) models, automatic speaker verification (ASV), and evaluation.

Main experiment themes:

- Time-domain embedding based spoofing detection.
- Time-domain embedding based gender classification.
- Male, female, and combined CM evaluation.
- Fixed-order score generation for reproducible CM/ASV/DCF alignment.
- Evaluation with EER, weighted EER, DET, DCF, t-DCF, a-DCF, and bootstrap confidence intervals.
- Visualization with UMAP, t-SNE, density plots, gender histograms, and correlation matrices.

## Repository Structure

| Path | File types | Purpose |
| --- | --- | --- |
| `ASV_utils/` | `.py` | ASV/CM helpers for enrollment handling, ECAPA-TDNN scoring, CM inference, data loading, DCF/t-DCF support, and thresholds. |
| `utils/` | `.py` | Shared training utilities, PyTorch datasets, losses, EER/DET metrics, plotting helpers, early stopping, and confidence intervals. |
| `Data/` | `.mat`, `.txt` | Processed time-domain embedding datasets and labels for `pmf_both` CM models and male-vs-female models database experiments. |
| `Deep_Learning_Codes/` | `.ipynb`, `.png`, `.py`, `.pkl` | Deep learning CM experiments, best-model notebooks, fixed-order runs, plots, attack-wise EER, and confidence intervals results. |
| `ML/` | `.ipynb`, `.py`, `.pkl` | Classical ML algorithms for: spoofed-vs-genuine EER experiments, gender classification, and histogram analysis. |
| `DCF/` | `.ipynb`, `.png`, `.pkl` | ECAPA-TDNN DCF notebooks, DCF figures, and combined ASV/CM t-DCF experiments. |
| `tDCF/` | `.ipynb`, `.pkl` | Total t-DCF and weighted EER notebooks. |
| `aDCF/` | `.ipynb`, `.py`, `.cfg` | a-DCF notebooks, fixed-order a-DCF/t-DCF analysis. |
| `time_embeddings_scores/` | `.csv` | Fixed-order train/dev/eval score files for male and female time-domain CM systems. |
| `Villalization_DM/` | `.ipynb`, `.png`, `.txt` | Visualization notebooks and figures for t-SNE, UMAP, density, gender, attack, and correlation analysis. |
| Repository root | `.md`, `.gitignore`, `LICENSE` | README, ignore rules, and MIT license. |

## Important Files

| File | Purpose |
| --- | --- |
| `ASV_utils/data_loading.py` | Loads male, female, and combined `.mat` embedding datasets and their label metadata. |
| `ASV_utils/ASV_my_functions.py` | Handles enrollment files, CM inference, ECAPA-TDNN ASV scoring, and combined ASV/CM inference. |
| `ASV_utils/config_thr.py` | Stores selected gender and spoofing thresholds used by notebooks. |
| `ASV_utils/dcf_my_functions.py` | Calculates ASV DCF from protocol files and cosine-similarity scores. |
| `ASV_utils/tdcf_functions.py` | Wraps t-DCF calculations for CM/ASV experiments. |
| `ASV_utils/My_CM_classes.py` | Defines the DNN-style countermeasure models used in deep learning notebooks. |
| `utils/DNN_functions.py` | Contains PyTorch training/testing loops, EER reporting, DET plotting, and checkpoint handling. |
| `utils/my_classes.py` | Defines PyTorch dataset classes and custom sampling utilities. |
| `utils/eval_metrics.py` | Implements EER, DET, constrained t-DCF, and unconstrained t-DCF metrics. |
| `utils/my_conf_inter.py` | Provides bootstrap confidence interval helpers. |
| `utils/AMSloss.py`, `utils/AdMSLoss_ver2.py`, `utils/OCSoftmaxloss.py` | Loss functions for AMSoftmax/AdMSoftmax and OCSoftmax models. |
| `aDCF/a_DCF_package/a_dcf/a_dcf.py` | a-DCF and cost model. |

## Data and Scores

Processed PMF/time-domain embedding data is stored under:

- `Data/pmf_both/not_normalize/male/`
- `Data/pmf_both/not_normalize/female/`
- `Data/male_vs_female_DB_models/`

The `.mat` files include embedding groups and matching labels for spoof status, attack type, file name, speaker ID, sex, and numeric labels. The suffixes `1_1` (training set), `2_1` (development set), and `3_1` (evaluation set) are used as train, dev, and eval split identifiers.

Fixed-order score CSVs are stored under:

- `time_embeddings_scores/fixed_order/male/`
- `time_embeddings_scores/fixed_order/female/`

together they are the gender-dependent CM system.

CSV columns:

```text
scores,labels,files
```

## Important Links

- **UMAP experiments data** (PMF-based embeddings of ASVspoof2019, gender and spoof/bonafide histograms, CM models based on PMF-based embeddings of spoof/bonafide histograms): [download the required zip files from Google Drive](https://drive.google.com/drive/folders/1udhCQOVr0SORGdQBCswUOiKdk-sqS4IC?usp=sharing)
- **Evaluation metrics only**: [GitHub — Evaluation-Measures-Audio-Deepfake-Detection](https://github.com/avishai111/Evaluation-Measures-Audio-Deepfake-Detection)
- **ECAPA-TDNN network**: [Hugging Face — speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

## Main Experiment Areas

| Path | Contents |
| --- | --- |
| `Deep_Learning_Codes/best_models/fixed order/models/` | Fixed-order best-model notebooks for male, female, and combined CM systems. Includes fully connected variants, original-ordering runs, plots, and attack-wise EER notebooks. |
| `Deep_Learning_Codes/conf_intervals/` | Confidence interval notebooks for CM, ASV, and t-DCF reporting. |
| `ML/pmf_both_spoofed_and_genuine_eer/` | Classical ML spoofed-vs-genuine EER experiments for all, male, and female splits. |
| `ML/pmf_both_classification_male_vs_female/` | Gender classification experiments and histogram-based male/female analysis. |
| `DCF/dcf_avg_score/dcf/dcf/` | Current ECAPA-TDNN DCF calculation notebooks for male, female, and all-speaker settings. |
| `DCF/dcf_avg_score/dcf/dcf_figures/` | Saved DCF, EER, and probability figures. |
| `DCF/dcf_avg_score/pmf_both_tdcf/` | Combined ASV/CM t-DCF experiments for the `pmf_both` setup. |
| `tDCF/` | Total t-DCF and weighted EER notebooks. |
| `aDCF/a_DCF_fixed_order/` | Fixed-order a-DCF and t-DCF notebooks. |
| `aDCF/a_DCF_package/` | Local installable helper package for a-DCF calculation. |
| `Villalization_DM/T_SNE_ALL_EMBEDDINGS/` | Current t-SNE figures for all embeddings and gender-specific views. |
| `Villalization_DM/Corr/` | Correlation matrix notebooks for time embeddings. |
| `Villalization_DM/Gender_UMAP_time_embeddings_based_gender_histograms/` | UMAP and histogram notebooks/figures for gender-related embedding behavior. |

## Key Metrics

- Equal Error Rate (EER)
- Weighted EER
- Detection Error Tradeoff (DET)
- Detection Cost Function (DCF)
- Tandem Detection Cost Function (t-DCF)
- Agnostic Detection Cost Function (a-DCF)
- Bootstrap confidence intervals
- Gender-specific and attack-wise performance

## Installation

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the main notebook dependencies:

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost
pip install torch torchaudio torchvision
pip install matplotlib seaborn plotly jupyter notebook ipykernel tqdm alive-progress
pip install speechbrain mlflow soundfile openpyxl joblib
pip install umap-learn pacmap
```

## Reproducibility Notes

- This is a research workspace, not a packaged Python library.
- Some notebooks use absolute paths from the original experiment environment and may need path updates.
- Fixed-order folders are intended for stable score alignment across CM, ASV, DCF, t-DCF, and a-DCF experiments.

## License

This project is released under the MIT License. See `LICENSE` for details.

## 📬 Contact

If you have questions, issues, or want to collaborate, feel free to reach out:

- 📧 Email: [Avishai Weizman](mailto:wavishay@post.bgu.ac.il)
- 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)
- 🎓 Google Scholar: [Avishai Weizman](https://scholar.google.com/citations?hl=iw&user=vWlnVpUAAAAJ)
- 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://www.linkedin.com/in/avishai-weizman/)
