"""Tune the CMSSW track-purity DNN working points on trackingNtuples.

Uses the score CMSSW computed (trk_mva), so the thresholds are exactly what the
TrackTorchClassifierFromSoA cuts will see. The ntuples must be made with the DNN cuts
switched off (qualityCutsPrompt/Displaced = -1), otherwise tracks below the loose cut
are already gone.

Per pT bin (inclusive / low pT < 5 GeV / high pT >= 5 GeV), per category (all / prompt
|dxy|<=0.5 / displaced |dxy|>0.5) and per sample set it finds the highest threshold that keeps
>= each target fraction of real (sim-matched) tracks, writes a summary, and makes a
prediction-distribution plot with the working points.

Example:
  python tune_working_points.py \
      --sample ttbar="/ceph/.../tt/*.root" --sample qcd="/ceph/.../qcd/*.root" \
      --out wp_pre4 --targets 0.995 0.99
"""
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from dataset import expand_input_files, track_labels

DUPLICATE_MERGE_ALGO = 2
HIGH_PT_TRIPLET_ALGO = 22
TREE = "trackingNtuple/tree"


def load_tracks(files, algo, max_events=None):
    """Return dict of flat numpy arrays (score, dxy, pt, is_real) for the selected tracks."""
    branches = ["trk_mva", "trk_dxy", "trk_pt", "trk_algo", "trk_originalAlgo", "trk_simTrkIdx"]
    out = {"score": [], "dxy": [], "pt": [], "is_real": []}
    n_events = 0
    for f in files:
        with uproot.open(f) as r:
            tree = r[TREE]
            stop = None if max_events is None else max(0, max_events - n_events)
            if stop == 0:
                break
            arr = tree.arrays(branches, entry_stop=stop, library="ak")
        n_events += len(arr)
        # Tracks rebuilt by the duplicate merger carry the duplicate classifier's score, not the DNN's.
        sel = (arr["trk_originalAlgo"] == algo) & (arr["trk_algo"] != DUPLICATE_MERGE_ALGO)
        out["score"].append(ak.to_numpy(ak.flatten(arr["trk_mva"][sel])))
        out["dxy"].append(ak.to_numpy(ak.flatten(arr["trk_dxy"][sel])))
        out["pt"].append(ak.to_numpy(ak.flatten(arr["trk_pt"][sel])))
        # Same real/fake definition as training (dataset.track_labels)
        is_real = ak.to_numpy(track_labels(arr["trk_simTrkIdx"]))
        out["is_real"].append(is_real[ak.to_numpy(ak.flatten(sel))])
    tracks = {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}
    tracks["n_events"] = n_events
    return tracks


def threshold_for_efficiency(scores_real, target):
    """Highest threshold t such that the fraction of real tracks with score >= t is >= target."""
    s = np.sort(scores_real)
    k = int(np.floor((1.0 - target) * len(s)))
    return float(s[min(k, len(s) - 1)])


def summarize(score, is_real, thr):
    keep = score >= thr
    n_real, n_fake = int(is_real.sum()), int((~is_real).sum())
    return {
        "threshold": thr,
        "efficiency": float(keep[is_real].mean()) if n_real else float("nan"),
        "fake_rejection": float((~keep[~is_real]).mean()) if n_fake else float("nan"),
        "fake_fraction_kept": float((~is_real[keep]).mean()) if keep.any() else float("nan"),
    }


def plot_distribution(score, is_real, wps, title, path):
    """Same layout as analysis.performance.plot_prediction_distribution, with several WP lines."""
    fake, real = score[~is_real], score[is_real]
    bins = np.linspace(0, 1, 51)
    styles = ["--", ":", "-."]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(fake, bins=bins, alpha=0.7, label=f"Fake Tracks ({len(fake):,})", color="red", edgecolor="black")
    axes[0].hist(real, bins=bins, alpha=0.7, label=f"Real Tracks ({len(real):,})", color="green", edgecolor="black")
    axes[1].hist([fake, real], bins=bins, alpha=0.7, label=["Fake Tracks", "Real Tracks"],
                 color=["red", "green"], edgecolor="black", stacked=True, density=True)
    for i, (target, wp) in enumerate(wps.items()):
        for ax in axes:
            ax.axvline(x=wp["threshold"], color="black", linestyle=styles[i % len(styles)], linewidth=2,
                       label=f"Threshold = {wp['threshold']:.4f} ({float(target)*100:.1f}% eff., "
                             f"{wp['fake_rejection']*100:.1f}% fakes rejected)")

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Predicted Probability (trk_mva)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"Distribution of Predicted Probabilities: {title}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Probability (trk_mva)", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Stacked Probability Distribution (Normalized)", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


PT_BINS = ("inclusive", "lowPt", "highPt")
CATS = ("all", "prompt", "displaced")


def cmssw_keep(t, cuts, dxy_thr, pt_thr):
    """CMSSW selection: per pT bin, pass if score >= prompt cut or (|dxy| > thr and score >= displaced cut).

    cuts: {pt_bin: (prompt_cut, displaced_cut)} with pt_bin "inclusive" or "lowPt"/"highPt".
    """
    disp = np.abs(t["dxy"]) > dxy_thr
    if "inclusive" in cuts:
        p = np.full(len(t["score"]), cuts["inclusive"][0])
        d = np.full(len(t["score"]), cuts["inclusive"][1])
    else:
        high = t["pt"] >= pt_thr
        p = np.where(high, cuts["highPt"][0], cuts["lowPt"][0])
        d = np.where(high, cuts["highPt"][1], cuts["lowPt"][1])
    return (t["score"] >= p) | (disp & (t["score"] >= d))


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="append", required=True, metavar="NAME=GLOB",
                        help="Sample name and ntuple glob; repeat for several samples")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.995, 0.99])
    parser.add_argument("--dxy-threshold", type=float, default=0.5, help="Same as dxyThreshold in CMSSW")
    parser.add_argument("--pt-threshold", type=float, default=5.0,
                        help="Same as ptThreshold in CMSSW: tracks with pT >= this use the high-pT cuts")
    parser.add_argument("--algo", type=int, default=HIGH_PT_TRIPLET_ALGO, help="trk_originalAlgo to select")
    parser.add_argument("--max-events", type=int, default=None, help="Per sample")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out", type=str, default="wp_results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {}
    for spec in args.sample:
        name, pattern = spec.split("=", 1)
        files = expand_input_files(pattern)
        if not files:
            raise ValueError(f"No files for {name}: {pattern}")
        logging.info(f"{name}: reading {len(files)} files")
        samples[name] = load_tracks(files, args.algo, args.max_events)
        logging.info(f"{name}: {samples[name]['n_events']} events, {len(samples[name]['score']):,} tracks")
        if len(samples[name]["score"]) == 0:
            logging.warning(f"{name}: no selected tracks (empty files or wrong --algo?)")
    if len(samples) > 1:
        combined = {k: np.concatenate([s[k] for s in samples.values()]) for k in ("score", "dxy", "pt", "is_real")}
        combined["n_events"] = sum(s["n_events"] for s in samples.values())
        samples["combined"] = combined

    results = {}
    for name, t in samples.items():
        disp = np.abs(t["dxy"]) > args.dxy_threshold
        high = t["pt"] >= args.pt_threshold
        all_tracks = np.ones_like(disp)
        results[name] = {"n_events": int(t["n_events"]), "pt_threshold": args.pt_threshold}
        for pt_bin, pt_mask in [("inclusive", all_tracks), ("lowPt", ~high), ("highPt", high)]:
            results[name][pt_bin] = {}
            for cat, mask in [("all", all_tracks), ("prompt", ~disp), ("displaced", disp)]:
                sel = pt_mask & mask
                score, is_real = t["score"][sel], t["is_real"][sel]
                entry = {"n_real": int(is_real.sum()), "n_fake": int((~is_real).sum()), "wps": {}}
                results[name][pt_bin][cat] = entry
                if entry["n_real"] == 0:
                    continue
                for target in args.targets:
                    entry["wps"][str(target)] = summarize(score, is_real,
                                                          threshold_for_efficiency(score[is_real], target))
                if not args.no_plots:
                    plot_distribution(score, is_real, entry["wps"], f"{name}, {pt_bin}, {cat}",
                                      out_dir / f"prediction_distribution_{name}_{pt_bin}_{cat}.png")

    # Overall efficiency with CMSSW's logic, for the pT-inclusive cuts and for the pT-dependent cuts
    for name, t in samples.items():
        res = results[name]
        for target in map(str, args.targets):
            cuts = {}
            for pt_bin in PT_BINS:
                wp_p = res[pt_bin]["prompt"]["wps"].get(target)
                wp_d = res[pt_bin]["displaced"]["wps"].get(target)
                if wp_p and wp_d:
                    cuts[pt_bin] = (wp_p["threshold"], wp_d["threshold"])
                    if wp_d["threshold"] > wp_p["threshold"]:
                        logging.warning(f"{name} {pt_bin} {target}: displaced cut > prompt cut; CMSSW's OR makes "
                                        f"the displaced cut effectively the prompt cut for displaced tracks")
            for label, c in [("inclusive", {k: v for k, v in cuts.items() if k == "inclusive"}),
                             ("ptDependent", {k: v for k, v in cuts.items() if k != "inclusive"})]:
                if len(c) != (1 if label == "inclusive" else 2):
                    continue
                keep = cmssw_keep(t, c, args.dxy_threshold, args.pt_threshold)
                entry = res.setdefault("cmssw_logic", {}).setdefault(label, {})[target] = {}
                for pt_bin, m in [("inclusive", np.ones_like(keep)), ("lowPt", t["pt"] < args.pt_threshold),
                                  ("highPt", t["pt"] >= args.pt_threshold)]:
                    r, f = t["is_real"] & m, ~t["is_real"] & m
                    entry[pt_bin] = {"efficiency": float(keep[r].mean()) if r.any() else float("nan"),
                                     "fake_rejection": float((~keep[f]).mean()) if f.any() else float("nan")}

    with open(out_dir / "working_points.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    for name, res in results.items():
        lines.append(f"\n=== {name} ({res['n_events']} events), pT threshold {args.pt_threshold} GeV")
        for pt_bin in PT_BINS:
            lines.append(f"  --- {pt_bin}")
            for cat in CATS:
                e = res[pt_bin][cat]
                lines.append(f"  {cat:9s} real {e['n_real']:>11,}  fake {e['n_fake']:>11,}")
                for target, wp in e["wps"].items():
                    lines.append(f"    {float(target)*100:5.1f}%: thr {wp['threshold']:.4f}  eff {wp['efficiency']:.4f}  "
                                 f"fake rej {wp['fake_rejection']:.4f}  fake frac kept {wp['fake_fraction_kept']:.4f}")
        lines.append("  --- CMSSW logic (prompt OR displaced), eff / fake rej per pT bin")
        for label, per_target in res.get("cmssw_logic", {}).items():
            for target, c in per_target.items():
                txt = "  ".join(f"{b} {c[b]['efficiency']:.4f}/{c[b]['fake_rejection']:.4f}" for b in PT_BINS)
                lines.append(f"    {label:11s} cuts at {float(target)*100:.1f}%: {txt}")
        # CMSSW vector order is (loose, tight, highPurity); current convention: 99.5%, 99%, 99%
        order = (0.995, 0.99, 0.99)
        lines.append(f"  trackTorchMVAClassifierQualityCuts.py (loose, tight, highPurity = 99.5%, 99%, 99%):")
        for pt_bin, suffix in [("inclusive", " (pT-inclusive)"), ("lowPt", ""), ("highPt", "")]:
            wps = res[pt_bin]
            if not all(str(t) in wps["prompt"]["wps"] and str(t) in wps["displaced"]["wps"] for t in order):
                continue
            # Round down so the rounded cut still keeps >= the target efficiency
            p = [np.floor(wps["prompt"]["wps"][str(t)]["threshold"] * 1000) / 1000 for t in order]
            d = [np.floor(wps["displaced"]["wps"][str(t)]["threshold"] * 1000) / 1000 for t in order]
            tag = "" if pt_bin == "inclusive" else pt_bin[0].upper() + pt_bin[1:]
            lines.append(f"    Prompt{tag}{suffix}:    HighPtTripletStep = cms.vdouble({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}),")
            lines.append(f"    Displaced{tag}{suffix}: HighPtTripletStep = cms.vdouble({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}),")
    text = "\n".join(lines)
    print(text)
    (out_dir / "working_points.txt").write_text(text + "\n")
    logging.info(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
