import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-Frequency Coupling (PAC)")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--run")
    p.add_argument("--low-bands", nargs=4, type=float, metavar=("l1", "l2", "l3", "l4"), help="Two low bands: l1 l2 l3 l4 (e.g., 4 8 8 12)")
    p.add_argument("--high-bands", nargs=4, type=float, metavar=("h1", "h2", "h3", "h4"), help="Two high bands: h1 h2 h3 h4 (e.g., 30 50 60 90)")
    p.add_argument("--metric", choices=["mi"], default="mi")
    p.add_argument("--surrogates", type=int, default=200)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def bandpass(data: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    b, a = butter(4, [lo / (sfreq / 2.0), hi / (sfreq / 2.0)], btype="band")
    return filtfilt(b, a, data, axis=-1)


def modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = 18) -> float:
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    digitized = np.digitize(phase, bins) - 1
    P = np.zeros(n_bins)
    for k in range(n_bins):
        P[k] = amp[digitized == k].mean() if np.any(digitized == k) else 0.0
    P = P / (P.sum() + 1e-12)
    H = -np.sum(P * np.log(P + 1e-12))
    H_max = np.log(n_bins)
    return (H_max - H) / H_max


def pac_one_sensor(x: np.ndarray, sfreq: float, low_band: Tuple[float, float], high_band: Tuple[float, float]) -> float:
    xl = bandpass(x, sfreq, low_band[0], low_band[1])
    xh = bandpass(x, sfreq, high_band[0], high_band[1])
    phase = np.angle(hilbert(xl))
    amp = np.abs(hilbert(xh))
    return modulation_index(phase, amp)


def time_shift_surrogates(x: np.ndarray, shift_samples: Sequence[int]) -> Sequence[np.ndarray]:
    n = x.shape[-1]
    return [np.roll(x, s % n, axis=-1) for s in shift_samples]


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load raw from BIDS
    bp = BIDSPath(root=args.bids_root, subject=args.subject.replace("sub-", ""), task=args.task, run=args.run)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=False, exclude="bads")
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]
    sfreq = raw.info["sfreq"]

    low_bands = [(args.low_bands[0], args.low_bands[1]), (args.low_bands[2], args.low_bands[3])]
    high_bands = [(args.high_bands[0], args.high_bands[1]), (args.high_bands[2], args.high_bands[3])]

    pac = np.zeros((len(low_bands), len(high_bands), len(ch_names)))
    pvals = np.ones_like(pac)

    rng = np.random.default_rng(0)
    shifts = rng.integers(low=int(0.2 * sfreq), high=int(0.8 * sfreq), size=args.surrogates)

    for i, lb in enumerate(low_bands):
        for j, hb in enumerate(high_bands):
            for c, ch in enumerate(ch_names):
                mi = pac_one_sensor(data[c], sfreq, lb, hb)
                # Surrogates: shift high-band signal
                sur = []
                for s in shifts:
                    xh = bandpass(data[c], sfreq, hb[0], hb[1])
                    xh_s = np.roll(xh, s)
                    amp_s = np.abs(hilbert(xh_s))
                    xl = bandpass(data[c], sfreq, lb[0], lb[1])
                    phase = np.angle(hilbert(xl))
                    sur.append(modulation_index(phase, amp_s))
                sur = np.asarray(sur)
                p = (np.sum(sur >= mi) + 1) / (len(sur) + 1)
                pac[i, j, c] = mi
                pvals[i, j, c] = p

    np.save(out / "pac.npy", pac)
    np.save(out / "pac_pvals.npy", pvals)

    # Simple comodulogram for first channel
    plt.figure(figsize=(4, 3))
    im = plt.imshow(pac[:, :, 0], origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Modulation Index")
    plt.xticks(range(len(high_bands)), [f"{a}-{b}" for a, b in high_bands])
    plt.yticks(range(len(low_bands)), [f"{a}-{b}" for a, b in low_bands])
    plt.xlabel("High band (Hz)")
    plt.ylabel("Low band (Hz)")
    plt.tight_layout()
    plt.savefig(out / "comodulogram_ch0.png", dpi=150)
    plt.close()

    print(f"Saved PAC arrays and figures to {out}")


if __name__ == "__main__":
    main()



