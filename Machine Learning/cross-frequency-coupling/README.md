## Cross-Frequency Coupling (PAC/CFC)

Compute phase–amplitude coupling (PAC) and related CFC metrics from EEG/MEG, with surrogate testing and comodulogram/topography outputs.

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python cfc_cli.py \
  --bids_root /path/to/bids \
  --subject sub-01 \
  --task memory \
  --low-bands 4 8 8 12 \
  --high-bands 30 50 60 90 \
  --metric mi \
  --surrogates 200 \
  --out_dir outputs/cfc_sub-01
```

### Outputs
- `pac.npy`: PAC values (low_band × high_band × sensors)
- `pac_pvals.npy`: p-values from surrogates
- Figures: comodulograms, topographies per band pair

### Notes
- Use appropriate filters and guard against edge artifacts. Surrogates control false positives.


