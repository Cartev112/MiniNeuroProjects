## Applied ML for Brain Data: Project Ideas by Increasing Complexity

This document lists implementable projects across EEG/MEG/ECoG/iEEG, fMRI/dMRI, LFP/spikes, and calcium imaging. Each item includes a concrete goal, suggested public datasets, core methods, and deliverables, plus stretch ideas. All projects are designed to be doable now with open-source tooling.

### How to use this list
- Start at your current level; ship a minimal version in days, then iterate.
- Prefer BIDS-formatted datasets and reproducible environments (conda/poetry + Docker) where feasible.
- Use the Dataset Index and Tooling References at the end to get started quickly.

---

## Tier 1 — Foundations and Baselines

1) EEG artifact detection and automatic quality reports
- Goal: Build an automated artifact detector (eye blinks, muscle, line noise) and a simple HTML report.
- Data: [EEG Motor Movement/Imagery (PhysioNet)](https://physionet.org/content/eegmmidb/1.0.0/), [TUH EEG](https://www.isip.piconepress.com/projects/tuh_eeg/), any OpenNeuro EEG.
- Methods/Tools: `mne`, `mne-bids`, band-pass/notch, ICA/SSP, spectral features, scikit-learn.
- Deliverables: CLI that ingests BIDS EEG and outputs per-recording QC metrics and plots.
- Stretch: Train a small CNN to detect artifacts from raw segments and compare vs ICA-based rules.

2) EEG power spectral density explorer
- Goal: Compute PSDs across channels/conditions and visualize topographies and bandpower statistics.
- Data: [MNE sample datasets](https://mne.tools/stable/overview/datasets_index.html).
- Methods/Tools: Welch/multitaper PSD, `mne.viz`, permutation tests.
- Deliverables: Notebook and functions to compute/save PSDs; publication-quality figures.
- Stretch: Mixed-effects model of bandpower vs subject covariates.

3) ERP pipeline for visual/auditory oddball
- Goal: End-to-end ERP extraction with preprocessing, epoching, baseline correction, and peak quantification.
- Data: [EEGBCI/MNE sample](https://mne.tools/stable/overview/datasets_index.html), any OpenNeuro oddball dataset.
- Methods/Tools: `mne`, robust averaging, peak detection.
- Deliverables: Reusable ERP scripts; CSV of component amplitudes/latencies.
- Stretch: Single-trial ERP decoding with logistic regression.

4) Sleep staging baseline on Sleep-EDF
- Goal: Train classical models for sleep stage classification from short EEG epochs.
- Data: [Sleep-EDF Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/).
- Methods/Tools: Feature extraction (bandpower, Hjorth, entropy), scikit-learn (SVM/RandomForest).
- Deliverables: Baseline benchmarks with cross-subject CV; confusion matrices.
- Stretch: Implement a small CNN and compare against classical features.

5) Motor imagery EEG classification with CSP + LDA
- Goal: Reproduce BCI classic baseline for left vs right hand imagery.
- Data: [BCI Competition IV 2a/2b](http://www.bbci.de/competition/iv/), [BNCI Horizon 2014-001](http://bnci-horizon-2020.eu/database/data-sets).
- Methods/Tools: `moabb` for datasets/evaluation; CSP, LDA.
- Deliverables: Script producing accuracy with subject-wise CV; MOABB-compatible results.
- Stretch: Riemannian geometry features (XDAWN, tangent space) with `pyriemann`.

6) BIDS conversion and validation for EEG
- Goal: Convert a raw EEG dataset to BIDS and validate it.
- Data: Any raw EEG; or test with MNE sample.
- Methods/Tools: `mne-bids`, `bids-validator`, `pybids`.
- Deliverables: A minimal BIDS converter CLI with logs and validation output.
- Stretch: Containerize as a BIDS App with `docker`.

7) Basic fMRI confounds and motion QC
- Goal: Compute fMRI image quality metrics and motion confounds; summarize per subject.
- Data: [OpenNeuro fMRI datasets](https://openneuro.org/), e.g., ds000030.
- Methods/Tools: `nilearn`, `nipype` or direct `fmriprep` outputs, `pandas`, plots.
- Deliverables: Notebook for extracting QC measures; HTML summary report.
- Stretch: Predict QC pass/fail with a classifier.

8) Resting-state functional connectivity matrices
- Goal: Extract ROI-based FC matrices and visualize networks.
- Data: OpenNeuro rs-fMRI (any with adequate TR), [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/).
- Methods/Tools: `nilearn` atlases, time series extraction, correlation/precision matrices.
- Deliverables: FC matrices in `npy`/`csv`; network plots; reproducible notebook.
- Stretch: Test reliability (ICC) across sessions.

9) MEG sensor-space time–frequency maps
- Goal: Compute time–frequency decompositions and average across trials.
- Data: [MNE sample MEG Faces](https://mne.tools/stable/overview/datasets_index.html).
- Methods/Tools: Morlet wavelets, multitaper, baseline normalization.
- Deliverables: Time–frequency plots; scripts for batch runs.
- Stretch: Decoding time-resolved category labels with logistic regression.

10) Structural MRI brain age baseline
- Goal: Predict chronological age from sMRI-derived features.
- Data: [IXI dataset](https://brain-development.org/ixi-dataset/), [OASIS-1/2](https://www.oasis-brains.org/).
- Methods/Tools: `freesurfer`/`fastsurfer` or `fsl` features; `xgboost`/`catboost`.
- Deliverables: Baseline MAE; feature importance plots.
- Stretch: Bias-corrected brain age delta estimation.

---

## Tier 2 — Strong Baselines and First Deep Models

11) Sleep staging with a compact CNN/CRNN
- Goal: Train a small deep model on Sleep-EDF and compare to classical baselines.
- Data: Sleep-EDF.
- Methods/Tools: `pytorch`, `torchmetrics`, focal loss, class balancing, augmentation.
- Deliverables: Reproducible training script; confusion matrix; per-stage F1.
- Stretch: Subject-independent evaluation and transfer learning across cohorts.

12) Braindecode benchmark on MOABB datasets
- Goal: Run standard convnets (Shallow/Deep4Net/EEGNet) on multiple EEG datasets.
- Data: Datasets via `moabb` (e.g., BNCI 2014-001, BCI IV 2a).
- Methods/Tools: `braindecode`, `mne`, `torch`.
- Deliverables: Table of accuracies across models/datasets; training curves.
- Stretch: Hyperparameter sweeps with Optuna and early stopping.

13) P300 speller detection
- Goal: Detect P300 responses in oddball paradigms.
- Data: BCI competition datasets with P300; `moabb` provides accessors.
- Methods/Tools: XDAWN + LDA; or compact CNN.
- Deliverables: AUC/accuracy; latency analysis.
- Stretch: Online-style evaluation with pseudo-real-time streams.

14) fMRI task decoding with simple classifiers
- Goal: Decode task conditions from GLM beta maps or from timeseries parcels.
- Data: OpenNeuro task fMRI; [HCP minimal task set](https://humanconnectome.org/).
- Methods/Tools: `nilearn.decoding`, logistic regression/SVM with nested CV.
- Deliverables: Balanced accuracy; activation visualizations.
- Stretch: Time-generalization matrices for event timing.

15) Graph features on rs-fMRI connectomes
- Goal: Predict phenotype (e.g., age/sex) from graph metrics.
- Data: ABIDE, ADHD-200, OpenNeuro rs-fMRI.
- Methods/Tools: `nilearn` connectomes; `bctpy`/`networkx`; scikit-learn.
- Deliverables: Feature table and model performance; permutation test.
- Stretch: Harmonize multi-site effects with ComBat.

16) MEG source vs sensor decoding comparison
- Goal: Compare decoding performance in sensor space vs minimum-norm source space.
- Data: MNE sample MEG.
- Methods/Tools: `mne` forward/inverse, regularized solvers; logistic regression.
- Deliverables: Performance comparison; code to compute source estimates.
- Stretch: Spatial searchlight decoding in source space.

17) Automated ICA component labeling for EEG
- Goal: Train a classifier to label ICA components as brain vs artifact.
- Data: Labeled components from public repos or self-label subset.
- Methods/Tools: Feature engineering on component spectra/maps; scikit-learn.
- Deliverables: Trained classifier; integration into preprocessing.
- Stretch: Small vision transformer on component topographies.

18) Diffusion MRI tractography features for age/sex prediction
- Goal: Build a baseline predictor from tract-level metrics.
- Data: [HCP 1200](https://humanconnectome.org/) (subset), [IXI dMRI](https://brain-development.org/ixi-dataset/).
- Methods/Tools: `dipy` for reconstruction; feature aggregation; scikit-learn.
- Deliverables: Feature extraction pipeline; baseline metrics.
- Stretch: Test different reconstruction models (DTI vs CSD) and compare.

---

## Tier 3 — Encoding/Decoding and Representation Analyses

19) EEG encoding model for auditory envelopes
- Goal: Predict EEG from stimulus envelopes (mTRF/encoding) and compute TRFs.
- Data: Natural speech EEG datasets (e.g., OpenNeuro speech EEG), MNE sample audio.
- Methods/Tools: `mTRF`-style linear models; ridge regression; time-lagged features.
- Deliverables: TRF plots and predictive R²; cross-validated across subjects.
- Stretch: Nonlinear encoding with shallow CNNs.

20) Visual encoding with fMRI using DNN features
- Goal: Map deep visual features to voxel responses.
- Data: [Algonauts](https://algonauts.csail.mit.edu/) or BOLD5000.
- Methods/Tools: `nilearn`, pretrained CNNs (ResNet/CLIP) for features; ridge regression.
- Deliverables: Voxelwise encoding R² maps; ROI summaries.
- Stretch: Layer-wise RSA vs ventral stream ROIs.

21) Representational Similarity Analysis (RSA) EEG/MEG vs DNNs
- Goal: Compare time-resolved neural RDMs to DNN RDMs.
- Data: MNE MEG sample or public MEG object datasets.
- Methods/Tools: Compute neural RDMs over time; DNN embeddings; Mantel/partial correlations.
- Deliverables: Time courses of brain–model similarity; significance testing.
- Stretch: Temporal generalization RSA.

22) Source localization benchmarking
- Goal: Benchmark dipole fits and distributed solvers across SNR/regularization.
- Data: Simulated + MNE sample MEG/EEG.
- Methods/Tools: `mne` forward/inverse, simulations, metrics on localization error.
- Deliverables: Reproducible benchmark notebook and plots.
- Stretch: Bayesian source imaging baseline.

23) ECoG high-gamma decoding for simple motor tasks
- Goal: Classify movement vs rest from high-gamma band features.
- Data: OpenNeuro iEEG/ECoG motor datasets (e.g., ds002904, ds003029).
- Methods/Tools: `mne`, `mne-bids`, bandpower features, scikit-learn.
- Deliverables: Decoding accuracy; electrode importance maps.
- Stretch: Compact CNN on raw iEEG with `braindecode`.

24) fMRI inter-subject correlation (ISC) on naturalistic stimuli
- Goal: Compute ISC during movie watching and relate to behavioral covariates.
- Data: OpenNeuro naturalistic fMRI (e.g., `studyforrest`).
- Methods/Tools: `nilearn` timeseries, ISC metrics, statistical tests.
- Deliverables: ROI ISC maps; group difference tests.
- Stretch: Hyperalignment to boost ISC across subjects.

25) EEG Riemannian geometry pipeline
- Goal: End-to-end pipeline using covariance-based features and tangent space mapping.
- Data: `moabb` datasets.
- Methods/Tools: `pyriemann`, `moabb`, scikit-learn.
- Deliverables: Benchmarked accuracy vs CSP baselines.
- Stretch: Domain adaptation via alignment on Riemannian manifolds.

26) Spike train GLM (LNP) encoding
- Goal: Fit GLM to spike trains given sensory or kinematic regressors.
- Data: DANDI/NWB spiking datasets; [Allen Brain Observatory](https://portal.brain-map.org/explore/circuits).
- Methods/Tools: `neo`, `elephant`, Poisson GLM; time-lagged design matrices.
- Deliverables: PSTH fits, predictive log-likelihood; code to load NWB.
- Stretch: Coupling filters and population models.

---

## Tier 4 — Transfer, Self-Supervision, and Graphs

27) Self-supervised pretraining for EEG with contrastive learning
- Goal: Pretrain a representation on unlabeled EEG, evaluate with linear probe.
- Data: TUH EEG, Sleep-EDF (unlabeled segments).
- Methods/Tools: `pytorch`, SimCLR/TS2Vec-style augmentations, `sklearn` probe.
- Deliverables: Ablation of augmentations; probe accuracy vs from-scratch.
- Stretch: Masked autoencoder for time series.

28) Cross-subject domain adaptation for motor imagery
- Goal: Train on source subjects; adapt to new subject with minimal labels.
- Data: MOABB motor imagery datasets.
- Methods/Tools: CORAL/MMD losses; domain-adversarial training.
- Deliverables: Pre/post-adaptation performance; code for adaptation loop.
- Stretch: Riemannian alignment + deep adaptation hybrid.

29) GNNs on structural/functional connectomes
- Goal: Classify diagnosis or predict behavior from brain graphs.
- Data: ABIDE, ADHD-200, HCP subset.
- Methods/Tools: `pytorch_geometric`, graph convs on FC/SC; stratified CV.
- Deliverables: GNN vs ML baselines; interpretability via edge saliency.
- Stretch: Multi-graph fusion (functional + structural) with multiplex GNNs.

30) ComBat harmonization and generalization study
- Goal: Quantify site effects; apply ComBat; test generalization.
- Data: Multi-site rs-fMRI (ABIDE) or sMRI (OASIS + IXI).
- Methods/Tools: `neuroHarmonize`/ComBat; scikit-learn; statistical tests.
- Deliverables: Pre/post harmonization model performance and effect size.
- Stretch: Bayesian ComBat with covariates and nonlinearity.

31) Normative modeling of cortical thickness
- Goal: Fit a Gaussian process regression normative model (age, sex covariates).
- Data: OASIS/IXI with FreeSurfer thickness.
- Methods/Tools: `PCNtoolkit` (normative modeling), GPR; outlier maps.
- Deliverables: Z-scores per region; visualization of deviations.
- Stretch: Multi-site hierarchical GPR.

32) MEG decoding with temporal generalization matrices
- Goal: Train decoders across time and test across all time points.
- Data: MEG faces MNE sample; or other MEG datasets.
- Methods/Tools: Sliding-window features; scikit-learn pipelines.
- Deliverables: Temporal generalization heatmaps; stats on diagonals/off-diagonals.
- Stretch: Source-space TGMs.

33) Multimodal fusion EEG+fNIRS
- Goal: Improve classification by fusing EEG and fNIRS features.
- Data: Public EEG-fNIRS datasets (OpenNeuro has several), or SEED-IV variants.
- Methods/Tools: Late/early fusion; feature selection; calibration strategies.
- Deliverables: Fusion vs unimodal comparison; fusion ablation study.
- Stretch: Cross-modal contrastive pretraining.

---

## Tier 5 — Generative, Causal, and Real-Time

34) Synthetic EEG generation with diffusion/VAEs for augmentation
- Goal: Train a generative model to synthesize realistic EEG segments.
- Data: Sleep-EDF or motor imagery EEG.
- Methods/Tools: 1D diffusion or VAE; quality metrics (PSD, Fréchet EEG Distance analogs).
- Deliverables: Augmentation study measuring downstream gains.
- Stretch: Conditional generation by class and subject.

35) Causal connectivity estimation and benchmarking
- Goal: Estimate directed connectivity (Granger, PCMCI) and validate on simulations.
- Data: Simulated multivariate signals + real EEG/MEG.
- Methods/Tools: `tigramite` (PCMCI), Granger VAR; false discovery control.
- Deliverables: Benchmark suite with ROC against ground truth simulations.
- Stretch: Nonlinear causal discovery on iEEG.

36) Closed-loop neurofeedback simulation environment
- Goal: Build a gym-like environment simulating SSVEP/SMR neurofeedback.
- Data: Pre-recorded EEG; simulated loops.
- Methods/Tools: `gymnasium`, simple RL agents; latency-aware evaluation.
- Deliverables: Reusable environment; baseline controllers.
- Stretch: Integrate real-time LSL stream for hardware-in-the-loop.

37) Real-time EEG decoding with LSL
- Goal: Online pipeline from LSL stream to classification and feedback.
- Data: Local EEG device or simulated LSL stream.
- Methods/Tools: `labstreaminglayer`, `mne-realtime`, lightweight model.
- Deliverables: Minimal end-to-end demo with latency measurements.
- Stretch: Adaptive decoding with online learning.

38) ECoG speech feature decoding
- Goal: Predict spectrotemporal features from high-gamma ECoG.
- Data: OpenNeuro ECoG speech datasets (e.g., ds003029, ds003542).
- Methods/Tools: Regularized linear models/CNNs; CCA; evaluation via correlation.
- Deliverables: Reconstruction quality vs baseline features.
- Stretch: Phoneme classification; robustness to electrode subsets.

39) Calcium imaging demixing and spike inference pipeline
- Goal: Implement CNMF-E or Suite2p-based pipeline; evaluate spike inference accuracy.
- Data: Open 2p datasets (DANDI, Allen Brain Observatory).
- Methods/Tools: `suite2p` or `CaImAn`; benchmarking against ground truth (if available).
- Deliverables: Processing scripts; ROI quality metrics; event rates.
- Stretch: Train a small denoising autoencoder for raw movies.

40) Privacy-preserving EEG analysis via differential privacy
- Goal: Train a classifier with DP-SGD and study accuracy–privacy tradeoffs.
- Data: Sleep-EDF or MOABB.
- Methods/Tools: `opacus` for DP-SGD; calibration of epsilon.
- Deliverables: Curves of performance vs privacy budget; reproducible config.
- Stretch: Combine DP with synthetic data augmentation.

---

## Tier 6 — Systems, Reproducibility, and Scale

41) Build a BIDS App for an EEG pipeline
- Goal: Containerize preprocessing + decoding as a BIDS App.
- Data: Any BIDS EEG dataset.
- Methods/Tools: `docker`, `bids-validator`, CLI with `argparse`.
- Deliverables: Container image; usage docs; example outputs.
- Stretch: Boutiques descriptor; integration tests in CI.

42) End-to-end fMRI predictive modeling template
- Goal: Template repo to go from BIDS to prediction with cross-validation.
- Data: OpenNeuro fMRI dataset.
- Methods/Tools: `fmriprep` outputs, `nilearn`, `sklearn`, `snakemake` or `dvc`.
- Deliverables: Cookiecutter-style template with docs.
- Stretch: Add hyperparameter optimization and model registry.

43) Federated learning simulation across EEG cohorts
- Goal: Simulate multi-site training with non-IID splits.
- Data: Combine multiple MOABB datasets.
- Methods/Tools: `flower` or `fedml`; simple CNN.
- Deliverables: Fed vs centralized performance; communication rounds analysis.
- Stretch: Personalization layers and FedProx-style algorithms.

44) Large-scale benchmarking harness for EEG models
- Goal: Unified harness to evaluate models across datasets with consistent metrics.
- Data: MOABB, TUH, Sleep-EDF.
- Methods/Tools: `moabb` loaders; Hydra configs; logging with `wandb`.
- Deliverables: Leaderboard-style results; reproducible scripts.
- Stretch: Add fairness/robustness metrics and corruptions.

45) Cross-modal alignment: CLIP-style EEG–stimulus embeddings
- Goal: Learn a shared embedding between EEG segments and visual/audio features.
- Data: EEG with time-locked stimuli; speech/vision features via pretrained models.
- Methods/Tools: Contrastive loss; `torch`; shallow projectors.
- Deliverables: Retrieval metrics (recall@k); t-SNE/UMAP visualizations.
- Stretch: Zero-shot classification via embeddings.

46) Connectome-based diffusion models for augmentation
- Goal: Generate plausible FC matrices to augment training.
- Data: ABIDE/ADHD-200 FCs.
- Methods/Tools: Graph diffusion/VAEs; validity checks (symmetry, PSD).
- Deliverables: Augment-and-train gains; sanity checks on generated graphs.
- Stretch: Conditional generation by phenotype.

47) Robustness and spurious correlation audits
- Goal: Systematically test EEG/fMRI models under corruptions and site shifts.
- Data: Multi-site datasets.
- Methods/Tools: Corruption benchmarks; leave-site-out CV; calibration metrics.
- Deliverables: Robustness report and recommended practices.
- Stretch: Adversarial augmentations for improved robustness.

48) Reproducible neuro-ML cookiecutter
- Goal: Cookiecutter template enforcing structure: data, conf, scripts, tests, docs.
- Data: N/A (template).
- Methods/Tools: `cookiecutter`, `pytest`, `pre-commit`, `ruff`/`flake8`.
- Deliverables: Template repo with CI; example EEG project initialized.
- Stretch: Add optional GPU training set-up with Lightning.

---

## Dataset Index (curated, mostly public/free)

- EEG
  - **Sleep-EDF**: [PhysioNet page](https://physionet.org/content/sleep-edfx/1.0.0/)
  - **EEG Motor Movement/Imagery**: [PhysioNet page](https://physionet.org/content/eegmmidb/1.0.0/)
  - **TUH EEG Corpus**: [TUH portal](https://www.isip.piconepress.com/projects/tuh_eeg/)
  - **BCI Competition IV (2a/2b)**: [Competition site](http://www.bbci.de/competition/iv/)
  - **BNCI Horizon datasets**: [BNCI portal](http://bnci-horizon-2020.eu/database/data-sets)
  - **DEAP (emotion EEG)**: [Dataset site](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
  - **MOABB access**: [MOABB docs](https://moabb.neurotechx.com/docs/)

- MEG
  - **MNE sample MEG (faces)**: [Dataset link](https://mne.tools/stable/overview/datasets_index.html)
  - **Cam-CAN MEG**: [Cam-CAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
  - **OMEGA**: [OMEGA MEG database](https://www.mcgill.ca/bic/resources/omega)

- fMRI
  - **OpenNeuro**: [OpenNeuro](https://openneuro.org/)
  - **ABIDE**: [ABIDE I/II](http://fcon_1000.projects.nitrc.org/indi/abide/)
  - **ADHD-200**: [Dataset site](http://fcon_1000.projects.nitrc.org/indi/adhd200/)
  - **BOLD5000**: [Project site](https://bold5000.github.io/)
  - **HCP**: [Human Connectome Project](https://humanconnectome.org/) (registration required)
  - **StudyForrest**: [Project](http://studyforrest.org/)

- Structural/diffusion MRI
  - **IXI**: [Dataset](https://brain-development.org/ixi-dataset/)
  - **OASIS**: [OASIS datasets](https://www.oasis-brains.org/)
  - **HCP dMRI**: [HCP](https://humanconnectome.org/)
  - **UK Biobank**: [UKB imaging](https://www.ukbiobank.ac.uk/) (application required)

- iEEG/ECoG, LFP, spikes, calcium
  - **OpenNeuro iEEG**: [Search iEEG](https://openneuro.org/search/modality:Electrocorticography)
  - **DANDI Archive (NWB)**: [DANDI](https://dandiarchive.org/)
  - **Allen Brain Observatory**: [Portal](https://portal.brain-map.org/explore/circuits)

---

## Tooling References (Python-centric)

- General neuroimaging/EEG
  - **MNE-Python**: [Docs](https://mne.tools/)
  - **MNE-BIDS**: [Docs](https://mne.tools/mne-bids/stable/index.html)
  - **MOABB**: [Docs](https://moabb.neurotechx.com/docs/)
  - **Braindecode**: [Docs](https://braindecode.org/)
  - **PyRiemann**: [Docs](https://pyriemann.readthedocs.io/)
  - **Nilearn**: [Docs](https://nilearn.github.io/)
  - **nipype**: [Docs](https://nipype.readthedocs.io/)
  - **BIDS Validator**: [GitHub](https://github.com/bids-standard/bids-validator)

- Spikes/LFP/NWB
  - **Neo**: [Docs](https://neo.readthedocs.io/)
  - **Elephant**: [Docs](https://elephant.readthedocs.io/)
  - **SpikeInterface**: [Docs](https://spikeinterface.readthedocs.io/)
  - **NWB (PyNWB)**: [Docs](https://pynwb.readthedocs.io/)

- Calcium imaging
  - **Suite2p**: [Docs](https://suite2p.readthedocs.io/)
  - **CaImAn**: [Docs](https://caiman.readthedocs.io/)

- MRI/dMRI/connectomics
  - **FSL**: [Docs](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
  - **FreeSurfer/FastSurfer**: [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/), [FastSurfer](https://fastsurfer.github.io/)
  - **DIPY**: [Docs](https://dipy.org/)
  - **bctpy**: [Docs](https://pypi.org/project/bctpy/)

- Modeling/ML
  - **scikit-learn**: [Docs](https://scikit-learn.org/)
  - **PyTorch**: [Docs](https://pytorch.org/)
  - **PyTorch Lightning**: [Docs](https://lightning.ai/docs/pytorch/stable/)
  - **Optuna**: [Docs](https://optuna.org/)
  - **PCNtoolkit (normative modeling)**: [Docs](https://pcntoolkit.readthedocs.io/)
  - **Opacus (DP-SGD)**: [Docs](https://opacus.ai/)
  - **PyTorch Geometric**: [Docs](https://pytorch-geometric.readthedocs.io/)

- Real-time/infra
  - **LabStreamingLayer (LSL)**: [GitHub](https://github.com/sccn/labstreaminglayer)
  - **MNE-Realtime**: [Docs](https://mne.tools/mne-realtime/stable/index.html)
  - **Snakemake**: [Docs](https://snakemake.readthedocs.io/)
  - **DVC**: [Docs](https://dvc.org/doc)
  - **Hydra**: [Docs](https://hydra.cc/docs/intro/)
  - **Docker**: [Docs](https://docs.docker.com/)

---

### Ethics and Reproducibility Notes
- Obtain necessary approvals where required; respect dataset licenses and privacy.
- Use fixed random seeds; report exact preprocessing and hyperparameters.
- Prefer cross-subject splits for generalization; report confidence intervals.

### Getting Started Template (quickstart)
```bash
conda create -n neuroml python=3.11 -y
conda activate neuroml
pip install mne mne-bids moabb braindecode nilearn scikit-learn pytorch-lightning torch torchvision torchaudio pyriemann optuna pynwb neo elephant spikeinterface dipy bctpy opacus pytorch-geometric -q
```

Then pick any project above, clone the relevant dataset with its Python loader (MOABB/Nilearn/MNE/DANDI), and iterate.

