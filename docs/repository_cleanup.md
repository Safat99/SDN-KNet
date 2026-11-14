# Repository cleanup notes

This cleanup groups our documents and figures while preserving the experiments and research findings. SDN-KNet did not reliably recover the injected delays. No model, loss, training setting, metric, dataset, or saved numerical result was changed.

## Inventory and layout

- `models/`, `training/`, `data/`, and `utils/` contain shared implementation code. `data/jena_climate_2009_2016.csv` is the original tracked dataset (420,551 rows, 15 columns).
- `experiments/` contains the configuration-driven experiments and tuning scripts. `configs/` contains three populated YAML files and the empty `real_world.yaml` placeholder.
- `analysis/plots.py` generates figures; `analysis/saver.py` writes histories, metrics, and arrays. `analysis/plots(old).py` preserves the earlier plotting implementation.
- `v2/` is a later experimental branch within the source tree, with five standalone experiment scripts, two synthetic-data helpers, and original console notes in `results.txt`. Its scripts reuse the main models, generators, and estimators. The `_v2_1` versions are separate variants, not confirmed replacements; none was merged or removed.
- `results/` retains local generated JSON, NPZ, and NPY outputs. These remain ignored. `results/figures/` contains the existing figures grouped by dataset and is no longer ignored, so the moved figures can be reviewed and tracked.
- `docs/report/` and `docs/presentation/` contain the supplied PDFs. The presentation contains unfinished placeholders; we preserved it as supplied rather than revising academic content.
- `slurm/` contains eight cluster-specific launch scripts. They assume submission from the repository root, installed modules, and a Conda environment named `sdn_tf`.
- `analysis/stats.py`, `data/generators/real_loader.py`, and `experiments/run_real.py` are empty placeholders and remain in place.
- Local `__pycache__/` directories remain ignored. The empty `reports/figures/` and `reports/` directories were removed after moving their figures.

## Path and dependency changes

We removed `.codex` from `requirements.txt`: it is not an imported Python dependency or an installable project requirement. We added unpinned `pandas` because the Jena loaders import it. Existing dependency versions remain unspecified.

Figure generators now save under `results/figures/`, with dataset-specific destinations in the experiment scripts. The older plotter default was updated too. The Jena GRU script now passes its output directory to `Plotter` directly rather than including a second directory in each figure name. Original figure filenames remain unchanged, including the Jena v2 script's misleading `non_linear_v2_knet.png` output name.

Configuration paths, the dataset path, numerical result paths, Python package locations, and SLURM launch targets remain unchanged. No model checkpoint loading/saving paths were found in the Python scripts.

We expanded `.gitignore` for virtual environments, Python tooling caches, build output, and common editor/OS files. Generated numerical results remain ignored; figures are explicitly allowed.

## Relocated files

All relocated files retain their original filenames and bytes. The 14 originally tracked files were moved with filesystem renames because `git mv` could not create `.git/index.lock` in this workspace's read-only Git directory. The changes are unstaged; Git can recognize these as renames when staged later. No commit or push was made.

The other 26 figures came from the previously ignored `reports/figures/` directory. They are now visible as untracked files under `results/figures/`; including them in a future commit is a separate staging decision.

| Original path | New path | Original Git status |
| --- | --- | --- |
| `CSCI_547_Project_Report_updated_final.pdf` | `docs/report/CSCI_547_Project_Report_updated_final.pdf` | Tracked |
| `csci-547-group5-presentation_v1.pdf` | `docs/presentation/csci-547-group5-presentation_v1.pdf` | Tracked |
| `jena_climate_gru_baseline.png` | `results/figures/jena_climate/jena_climate_gru_baseline.png` | Tracked |
| `jena_climate_masking_vs_filled.png` | `results/figures/jena_climate/jena_climate_masking_vs_filled.png` | Tracked |
| `jena_climate_raw.png` | `results/figures/jena_climate/jena_climate_raw.png` | Tracked |
| `knet_after_training_on_linear_model.png` | `results/figures/linear/knet_after_training_on_linear_model.png` | Tracked |
| `knet_forward_pass_on_linear_model.png` | `results/figures/linear/knet_forward_pass_on_linear_model.png` | Tracked |
| `linear_signal_with_corruption1.png` | `results/figures/linear/linear_signal_with_corruption1.png` | Tracked |
| `linear_signal_with_corruption_with_GRU_baseline.png` | `results/figures/linear/linear_signal_with_corruption_with_GRU_baseline.png` | Tracked |
| `linear_signal_with_corruption_with_GRU_baseline_with_test_set.png` | `results/figures/linear/linear_signal_with_corruption_with_GRU_baseline_with_test_set.png` | Tracked |
| `linear_signal_with_corruption_with_cho_Encoder.png` | `results/figures/linear/linear_signal_with_corruption_with_cho_Encoder.png` | Tracked |
| `linear_signal_with_corruption_with_cho_Encoder_test_set.png` | `results/figures/linear/linear_signal_with_corruption_with_cho_Encoder_test_set.png` | Tracked |
| `non_linear_v2_gru.png` | `results/figures/mackey_glass/non_linear_v2_gru.png` | Tracked |
| `raw_mackey_glass_signal.png` | `results/figures/mackey_glass/raw_mackey_glass_signal.png` | Tracked |
| `reports/figures/gru_linear_d0_n0.0_s0_estimation.png` | `results/figures/linear/gru_linear_d0_n0.0_s0_estimation.png` | Ignored local figure |
| `reports/figures/gru_linear_d0_n0.0_s0_full.png` | `results/figures/linear/gru_linear_d0_n0.0_s0_full.png` | Ignored local figure |
| `reports/figures/gru_linear_d0_n0.0_s0_loss.png` | `results/figures/linear/gru_linear_d0_n0.0_s0_loss.png` | Ignored local figure |
| `reports/figures/gru_linear_d0_n0.0_s0_mae.png` | `results/figures/linear/gru_linear_d0_n0.0_s0_mae.png` | Ignored local figure |
| `reports/figures/gru_linear_d0_n0.0_s0_rmse.png` | `results/figures/linear/gru_linear_d0_n0.0_s0_rmse.png` | Ignored local figure |
| `reports/figures/gru_linear_d10_n1.0_s2_estimation.png` | `results/figures/linear/gru_linear_d10_n1.0_s2_estimation.png` | Ignored local figure |
| `reports/figures/gru_linear_d10_n1.0_s2_full.png` | `results/figures/linear/gru_linear_d10_n1.0_s2_full.png` | Ignored local figure |
| `reports/figures/gru_linear_d10_n1.0_s2_loss.png` | `results/figures/linear/gru_linear_d10_n1.0_s2_loss.png` | Ignored local figure |
| `reports/figures/gru_linear_d10_n1.0_s2_mae.png` | `results/figures/linear/gru_linear_d10_n1.0_s2_mae.png` | Ignored local figure |
| `reports/figures/gru_linear_d10_n1.0_s2_rmse.png` | `results/figures/linear/gru_linear_d10_n1.0_s2_rmse.png` | Ignored local figure |
| `reports/figures/gru_mg_d10_n1.0_s2_estimation.png` | `results/figures/mackey_glass/gru_mg_d10_n1.0_s2_estimation.png` | Ignored local figure |
| `reports/figures/gru_mg_d10_n1.0_s2_full.png` | `results/figures/mackey_glass/gru_mg_d10_n1.0_s2_full.png` | Ignored local figure |
| `reports/figures/gru_mg_d10_n1.0_s2_loss.png` | `results/figures/mackey_glass/gru_mg_d10_n1.0_s2_loss.png` | Ignored local figure |
| `reports/figures/gru_mg_d10_n1.0_s2_mae.png` | `results/figures/mackey_glass/gru_mg_d10_n1.0_s2_mae.png` | Ignored local figure |
| `reports/figures/gru_mg_d10_n1.0_s2_rmse.png` | `results/figures/mackey_glass/gru_mg_d10_n1.0_s2_rmse.png` | Ignored local figure |
| `reports/figures/kf_linear_d0_n1.0_s2_estimation.png` | `results/figures/linear/kf_linear_d0_n1.0_s2_estimation.png` | Ignored local figure |
| `reports/figures/kf_linear_d0_n1.0_s2_full.png` | `results/figures/linear/kf_linear_d0_n1.0_s2_full.png` | Ignored local figure |
| `reports/figures/kf_linear_d100_n0.0_s2_estimation.png` | `results/figures/linear/kf_linear_d100_n0.0_s2_estimation.png` | Ignored local figure |
| `reports/figures/kf_linear_d100_n0.0_s2_full.png` | `results/figures/linear/kf_linear_d100_n0.0_s2_full.png` | Ignored local figure |
| `reports/figures/kf_linear_d100_n5.0_s2_estimation.png` | `results/figures/linear/kf_linear_d100_n5.0_s2_estimation.png` | Ignored local figure |
| `reports/figures/kf_linear_d100_n5.0_s2_full.png` | `results/figures/linear/kf_linear_d100_n5.0_s2_full.png` | Ignored local figure |
| `reports/figures/kf_linear_d10_n1.0_s2_estimation.png` | `results/figures/linear/kf_linear_d10_n1.0_s2_estimation.png` | Ignored local figure |
| `reports/figures/kf_linear_d10_n1.0_s2_full.png` | `results/figures/linear/kf_linear_d10_n1.0_s2_full.png` | Ignored local figure |
| `reports/figures/kf_linear_d10_n1.0_s2_with_em_estimation.png` | `results/figures/linear/kf_linear_d10_n1.0_s2_with_em_estimation.png` | Ignored local figure |
| `reports/figures/kf_linear_d10_n1.0_s2_with_em_full.png` | `results/figures/linear/kf_linear_d10_n1.0_s2_with_em_full.png` | Ignored local figure |
| `reports/figures/ssl_gru_linear_d2_n0.0_s0_residual.png` | `results/figures/linear/ssl_gru_linear_d2_n0.0_s0_residual.png` | Ignored local figure |

## Validation and remaining issues

- All 38 Python files parse successfully; all eight SLURM scripts pass `bash -n`.
- All three populated YAML configurations parse; `configs/real_world.yaml` is empty.
- Local Markdown document/image links resolve, and the Jena configuration's dataset path and temperature column exist.
- SHA-256 checks confirm all 40 relocated files are unchanged. The tracked source copies of the 14 moved files match Git HEAD. The dataset and feedback PDF also match Git HEAD.
- Python AST comparison confirms changes are confined to plotting/output routing and directory creation, with model and training logic preserved.
- A headless plotting smoke check exercises output creation with small synthetic arrays; it is not a reproduction of research results.
- TensorFlow, pykalman, and Optuna are unavailable in the current Python environment. We did not install dependencies or run training, GPU jobs, or SLURM jobs. Runtime compatibility and full reproduction remain unverified.
- Four existing scripts (`run_kalmannet_linear.py`, `run_kalmannet_linear_final.py`, `run_knet_jena.py`, and `run_knet_mg.py`) import `apply_delay_compensation`, but its definition in `utils/delay_estimation.py` is commented out. We did not substitute `compensate_delay`, since choosing delay semantics would go beyond organization and could change the experiment.
- `slurm/run_linear.slurm` creates `slurm_logs/linear` while its output directives use `slurm_logs/linear_synthetic_ssm/updated`. SLURM opens log files before script execution, so required log directories must exist before submission. The Mackey–Glass tuning launcher has the same mismatch for its `updated/` subdirectory. Other scripts also create their log directories inside the job. We left cluster execution settings unchanged.
- The linear SLURM array has 120 tasks, whereas its EM selector only reaches the second value starting at index 120. The separate KF+EM launcher exists; the array logic was preserved rather than changing the experimental grid.
- Both linear v2 variants write the same figure filename. Running either can overwrite that output, as before. This cleanup does not establish a unique run-output scheme.

## Decisions before deletion or archiving

`Group5Report (1).pdf` contains grading/feedback. It remains at its original location, unchanged and unlinked from the research documents. Please decide how to handle it before publishing further changes. It is already tracked; moving or deleting it does not remove it from Git history.

The earlier plotter, empty placeholders, separate linear KalmanNet scripts, and v2 variants are retained. Their overlap alone does not establish that they are redundant. No deletion or archive move was made. No license was added.

## Final repository tree

The tree below lists project files and local numerical outputs, excluding Git internals and Python caches.

```text
sdn-knet/
├── analysis/
│   ├── plots(old).py
│   ├── plots.py
│   ├── saver.py
│   └── stats.py
├── configs/
│   ├── jena.yaml
│   ├── linear.yaml
│   ├── mackey_glass.yaml
│   └── real_world.yaml
├── data/
│   ├── corruption/
│   │   ├── delay.py
│   │   └── noise.py
│   ├── generators/
│   │   ├── linear_ssm.py
│   │   ├── mackey_glass.py
│   │   └── real_loader.py
│   ├── raw/
│   ├── jena_climate_2009_2016.csv
│   └── pipeline.py
├── docs/
│   ├── presentation/
│   │   └── csci-547-group5-presentation_v1.pdf
│   ├── report/
│   │   └── CSCI_547_Project_Report_updated_final.pdf
│   └── repository_cleanup.md
├── experiments/
│   ├── check_mg_signal.py
│   ├── run_gru_jena.py
│   ├── run_kalmannet_linear.py
│   ├── run_kalmannet_linear_final.py
│   ├── run_knet_jena.py
│   ├── run_knet_mg.py
│   ├── run_linear.py
│   ├── run_linear_ssl_encoder.py
│   ├── run_mg.py
│   ├── run_real.py
│   ├── tune_linear_gru.py
│   └── tune_mg_gru.py
├── models/
│   ├── gru_baseline.py
│   ├── gru_encoder_cho.py
│   ├── kalmannet_wrapper.py
│   └── kf_baselines.py
├── reports/
│   └── figures/
│       └── self-supervised-settings/
│           ├── ssl_gru_linear_d2_n0.0_s0_estimation.png
│           ├── ssl_gru_linear_d2_n0.0_s0_loss.png
│           ├── ssl_gru_linear_d2_n0.0_s0_mae.png
│           └── ssl_gru_linear_d2_n0.0_s0_rmse.png
├── results/
│   ├── figures/
│   │   ├── jena_climate/
│   │   │   ├── jena_climate_gru_baseline.png
│   │   │   ├── jena_climate_masking_vs_filled.png
│   │   │   └── jena_climate_raw.png
│   │   ├── linear/
│   │   │   ├── gru_linear_d0_n0.0_s0_estimation.png
│   │   │   ├── gru_linear_d0_n0.0_s0_full.png
│   │   │   ├── gru_linear_d0_n0.0_s0_loss.png
│   │   │   ├── gru_linear_d0_n0.0_s0_mae.png
│   │   │   ├── gru_linear_d0_n0.0_s0_rmse.png
│   │   │   ├── gru_linear_d10_n1.0_s2_estimation.png
│   │   │   ├── gru_linear_d10_n1.0_s2_full.png
│   │   │   ├── gru_linear_d10_n1.0_s2_loss.png
│   │   │   ├── gru_linear_d10_n1.0_s2_mae.png
│   │   │   ├── gru_linear_d10_n1.0_s2_rmse.png
│   │   │   ├── kf_linear_d0_n1.0_s2_estimation.png
│   │   │   ├── kf_linear_d0_n1.0_s2_full.png
│   │   │   ├── kf_linear_d100_n0.0_s2_estimation.png
│   │   │   ├── kf_linear_d100_n0.0_s2_full.png
│   │   │   ├── kf_linear_d100_n5.0_s2_estimation.png
│   │   │   ├── kf_linear_d100_n5.0_s2_full.png
│   │   │   ├── kf_linear_d10_n1.0_s2_estimation.png
│   │   │   ├── kf_linear_d10_n1.0_s2_full.png
│   │   │   ├── kf_linear_d10_n1.0_s2_with_em_estimation.png
│   │   │   ├── kf_linear_d10_n1.0_s2_with_em_full.png
│   │   │   ├── knet_after_training_on_linear_model.png
│   │   │   ├── knet_forward_pass_on_linear_model.png
│   │   │   ├── linear_signal_with_corruption1.png
│   │   │   ├── linear_signal_with_corruption_with_GRU_baseline.png
│   │   │   ├── linear_signal_with_corruption_with_GRU_baseline_with_test_set.png
│   │   │   ├── linear_signal_with_corruption_with_cho_Encoder.png
│   │   │   ├── linear_signal_with_corruption_with_cho_Encoder_test_set.png
│   │   │   └── ssl_gru_linear_d2_n0.0_s0_residual.png
│   │   └── mackey_glass/
│   │       ├── gru_mg_d10_n1.0_s2_estimation.png
│   │       ├── gru_mg_d10_n1.0_s2_full.png
│   │       ├── gru_mg_d10_n1.0_s2_loss.png
│   │       ├── gru_mg_d10_n1.0_s2_mae.png
│   │       ├── gru_mg_d10_n1.0_s2_rmse.png
│   │       ├── non_linear_v2_gru.png
│   │       └── raw_mackey_glass_signal.png
│   ├── gru_linear_d0_n0.0_s0_estimations/
│   │   ├── x_hat.npy
│   │   ├── x_true.npy
│   │   └── y_input.npy
│   ├── self-supervised-settings/
│   │   ├── ssl_gru_linear_d2_n0.0_s0_estimations.npz
│   │   ├── ssl_gru_linear_d2_n0.0_s0_history.json
│   │   └── ssl_gru_linear_d2_n0.0_s0_metrics.json
│   ├── gru_linear_d0_n0.0_s0_estimations.npz
│   ├── gru_linear_d0_n0.0_s0_history.json
│   ├── gru_linear_d0_n0.0_s0_metrics.json
│   ├── gru_linear_d10_n1.0_s2_estimations.npz
│   ├── gru_linear_d10_n1.0_s2_history.json
│   ├── gru_linear_d10_n1.0_s2_metrics.json
│   ├── gru_mg_d10_n1.0_s2_estimations.npz
│   ├── gru_mg_d10_n1.0_s2_history.json
│   ├── gru_mg_d10_n1.0_s2_metrics.json
│   ├── kf_linear_d0_n1.0_s2_estimations.npz
│   ├── kf_linear_d0_n1.0_s2_metrics.json
│   ├── kf_linear_d100_n0.0_s2_estimations.npz
│   ├── kf_linear_d100_n0.0_s2_metrics.json
│   ├── kf_linear_d100_n5.0_s2_estimations.npz
│   ├── kf_linear_d100_n5.0_s2_metrics.json
│   ├── kf_linear_d10_n1.0_s2_estimations.npz
│   ├── kf_linear_d10_n1.0_s2_metrics.json
│   ├── kf_linear_d10_n1.0_s2_with_em_estimations.npz
│   └── kf_linear_d10_n1.0_s2_with_em_metrics.json
├── slurm/
│   ├── run_knet_jena.slurm
│   ├── run_knet_linear.slurm
│   ├── run_knet_mg.slurm
│   ├── run_linear.slurm
│   ├── run_linear_kf_em_only.slurm
│   ├── run_mg_experiments.slurm
│   ├── tune_gru_linear.slurm
│   └── tune_mg_gru.slurm
├── training/
│   ├── losses.py
│   ├── metrics.py
│   └── trainer.py
├── utils/
│   ├── delay_estimation.py
│   └── noise_estimation.py
├── v2/
│   ├── data/
│   │   ├── corrupt.py
│   │   └── simple_linear.py
│   ├── experiments/
│   │   ├── run_jena_climate_v2.py
│   │   ├── run_jena_climate_v2_1.py
│   │   ├── run_knet_non_linear_v2.py
│   │   ├── run_linear_v2.py
│   │   └── run_linear_v2_1.py
│   └── results.txt
├── Group5Report (1).pdf
├── README.md
└── requirements.txt
```
