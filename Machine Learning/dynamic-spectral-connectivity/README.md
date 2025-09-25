## Dynamic Spectral Connectivity

Sliding-window spectral connectivity (coherence, wPLI, PLV) for EEG/MEG with optional state discovery via k-means. Saves time-resolved edge matrices, state labels, dwell/transition metrics, and figures.

### Install

```bash
pip install -r requirements.txt
```

### Run

Example (EEG BIDS, coherence and wPLI in alpha band):

```bash
python dsc_cli.py \
  --bids_root /path/to/bids \
  --subject sub-01 \
  --task rest \
  --band 8 12 \
  --metric coh wpli \
  --win_ms 2000 --step_ms 250 \
  --states 4 \
  --out_dir outputs/dsc_sub-01
```

### Outputs
- `edges_<metric>.npy`: array (windows × edges) per metric
- `state_labels.npy`: per-window state labels (if clustering)
- `state_metrics.csv`: dwell times, transition rates
- Figures: state connectivity maps, time courses

### Notes
- Use wPLI/imag coh to mitigate volume conduction.
- Choose window long enough for band resolution; step for temporal granularity.


