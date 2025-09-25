import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic Spectral Connectivity (EEG/MEG)")
    p.add_argument("--bids_root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--run")
    p.add_argument("--band", nargs=2, type=float, metavar=("FMIN", "FMAX"), default=[8.0, 12.0])
    p.add_argument("--metric", nargs="+", choices=["coh", "imag", "wpli", "plv"], default=["coh"])  # imag=imag coherence
    p.add_argument("--win_ms", type=int, default=2000)
    p.add_argument("--step_ms", type=int, default=250)
    p.add_argument("--states", type=int, default=0, help="If >0, run k-means clustering with this many states")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_epochs(bids_root: str, subject: str, task: str, run: str | None) -> mne.BaseEpochs:
    bp = BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), task=task, run=run)
    raw = read_raw_bids(bp, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=False, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=0.8, baseline=(-0.2, 0.0), picks=picks, preload=True, verbose=False)
    return epochs


def sliding_windows(n_times: int, win_samp: int, step_samp: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start + win_samp <= n_times:
        yield start, start + win_samp
        start += step_samp


def compute_connectivity(epochs: mne.BaseEpochs, fmin: float, fmax: float, metric: str, win_samp: int, step_samp: int) -> tuple[np.ndarray, list[str]]:
    data = epochs.get_data()  # (N, C, T)
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    # Concatenate epochs along time for simple sliding windows
    X = np.concatenate(data, axis=2)  # (N, C, T) -> (C, N*T) after transpose
    X = np.transpose(X, (1, 0, 2)).reshape(len(ch_names), -1)

    con_list = []
    for beg, end in sliding_windows(X.shape[1], win_samp, step_samp):
        seg = X[:, beg:end]
        seg_raw = mne.io.RawArray(seg, mne.create_info(ch_names, sfreq, ch_types=mne.pick_info(epochs.info, mne.pick_types(epochs.info, meg=True, eeg=True))["ch_names"]))
        fmin_, fmax_ = fmin, fmax
        con = mne.connectivity.spectral_connectivity_raw(
            seg_raw,
            method={
                "coh": "coh",
                "imag": "imcoh",
                "wpli": "wpli2_debiased",
                "plv": "plv",
            }[metric],
            fmin=fmin_, fmax=fmax_, faverage=True, mt_adaptive=False, n_jobs=1, verbose=False,
        )[0]  # shape (n_nodes, n_nodes, n_freqs=1)
        # Upper triangle to vector
        iu = np.triu_indices_from(con[:, :, 0], k=1)
        edges = con[:, :, 0][iu]
        con_list.append(edges)

    edges_arr = np.asarray(con_list)  # (windows, edges)
    labels = [f"{a}-{b}" for a, b in zip(*np.triu_indices(len(ch_names), 1))]
    return edges_arr, labels


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    epochs = load_epochs(args.bids_root, args.subject, args.task, args.run)

    win_samp = int(args.win_ms * 1e-3 * epochs.info["sfreq"])
    step_samp = int(args.step_ms * 1e-3 * epochs.info["sfreq"])

    all_metrics = {}
    for m in args.metric:
        edges, labels = compute_connectivity(epochs, args.band[0], args.band[1], m, win_samp, step_samp)
        np.save(out / f"edges_{m}.npy", edges)
        all_metrics[m] = edges

    # Optional state clustering using concatenated metrics
    if args.states and args.states > 0:
        Z = np.concatenate(list(all_metrics.values()), axis=1)
        Z = StandardScaler().fit_transform(Z)
        kmeans = KMeans(n_clusters=args.states, random_state=0, n_init=10)
        labels_state = kmeans.fit_predict(Z)
        np.save(out / "state_labels.npy", labels_state)

        # Simple dwell/transition metrics
        dwell = []
        cur = labels_state[0]; run = 1
        runs = []
        for i in range(1, len(labels_state)):
            if labels_state[i] == cur:
                run += 1
            else:
                runs.append((cur, run)); cur = labels_state[i]; run = 1
        runs.append((cur, run))
        from collections import Counter
        counts = Counter([r[0] for r in runs])
        dwell = {s: np.mean([r for (st, r) in runs if st == s]) for s in counts}
        trans_rate = (labels_state[1:] != labels_state[:-1]).mean()
        with open(out / "state_metrics.csv", "w") as f:
            f.write("state,dwell_windows\n")
            for s, d in dwell.items():
                f.write(f"{s},{d}\n")
            f.write(f"transitions_per_step,{trans_rate}\n")

        # Quick figure: state time series
        plt.figure(figsize=(10, 2))
        plt.plot(labels_state, drawstyle="steps-post")
        plt.yticks(range(args.states))
        plt.xlabel("Window")
        plt.ylabel("State")
        plt.tight_layout()
        plt.savefig(out / "states_timeseries.png", dpi=150)
        plt.close()

    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()



