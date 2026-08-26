#!/usr/bin/env python3
"""Build PhysWM paper tables and figures from experiment JSON artifacts.

Usage:
    python paper_package/build_paper_assets.py \
        --runs-root /workspace/physwm-artifacts/runs

Only the explicitly selected, auditable experiment groups below are used.
Pre-fix Fetch results are deliberately excluded from quantitative figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "navy": "#172033",
    "blue": "#3157d5",
    "cyan": "#12a4a6",
    "amber": "#d97706",
    "gray": "#8a94a6",
    "red": "#c2413b",
}


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.facecolor": "white",
    })


def architecture_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, color, sub=""):
        from matplotlib.patches import FancyBboxPatch
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                               linewidth=1.5, edgecolor=color, facecolor=color + "14")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.60, label, ha="center", va="center",
                weight="bold", color=COLORS["navy"])
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                    fontsize=7.5, color="#5c687c")

    def arrow(x1, y1, x2, y2, label="", color=None):
        color = color or COLORS["navy"]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=color))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                    ha="center", fontsize=7.5, color=color)

    box(0.2, 2.0, 1.4, 1.0, "History", COLORS["gray"], r"$o_{\leq t}, a_{\leq t}$")
    box(2.0, 2.0, 1.5, 1.0, "Encoder", COLORS["blue"], r"$E_\phi$")
    box(3.9, 2.0, 1.7, 1.0, "Predictor", COLORS["blue"], r"$P_\psi(z_{\leq t},a_{\leq t})$")
    box(6.0, 2.0, 1.3, 1.0, "Shared latent", COLORS["cyan"], r"$\hat z_t$")
    box(7.9, 3.35, 1.5, 1.0, "Neural decoder", COLORS["blue"], r"$D_\omega$")
    box(10.0, 3.35, 1.5, 1.0, "Prediction A", COLORS["blue"], r"$\hat s^A_{t+1}$")
    box(7.9, 0.55, 1.5, 1.0, "Linear probe", COLORS["cyan"], r"$\rho_\xi \rightarrow \hat\theta$")
    box(10.0, 0.55, 1.5, 1.0, "Frozen equations", COLORS["amber"], r"$S(s_t,a_t,\hat\theta)$")

    arrow(1.6, 2.5, 2.0, 2.5)
    arrow(3.5, 2.5, 3.9, 2.5)
    arrow(5.6, 2.5, 6.0, 2.5)
    arrow(7.3, 2.65, 7.9, 3.75)
    arrow(9.4, 3.85, 10.0, 3.85)
    arrow(7.3, 2.3, 7.9, 1.05)
    arrow(9.4, 1.05, 10.0, 1.05)
    arrow(10.75, 3.35, 10.75, 1.55, "stop-gradient teacher", COLORS["amber"])
    ax.text(10.75, 4.72, r"$\mathcal{L}_A=\|\hat s^A_{t+1}-s_{t+1}\|^2$",
            ha="center", color=COLORS["blue"], fontsize=9)
    ax.text(10.75, 0.12, r"$\mathcal{L}_B=\|S(s_t,a_t,\hat\theta)-\mathrm{sg}(\hat s^A_{t+1})\|^2$",
            ha="center", color=COLORS["amber"], fontsize=9)
    ax.set_title("PhysWM: grounding a predictive world model through known physical equations",
                 weight="bold", color=COLORS["navy"], pad=4)
    save_figure(fig, out, "fig1_architecture")


def pokeworld_figures(runs: Path, out: Path) -> list[dict]:
    decod = [load(runs / "pilot-2048ep" / f"A_pilot_2048ep_seed{s}.json") for s in range(3)]
    functional = [load(runs / "functional-2048" / f"func2048_seed{s}.json") for s in range(3)]

    # Functional one-step and seven-step evaluation.
    sources = ["probe", "true", "shuffled", "nominal"]
    labels = ["Inferred", "True", "Shuffled", "Nominal"]
    vals = {k: [r["substitution"][k] for r in functional] for k in sources}
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    x = np.arange(len(sources))
    means = [mean(vals[k]) for k in sources]
    errs = [stdev(vals[k]) for k in sources]
    colors = [COLORS["cyan"], COLORS["blue"], COLORS["red"], COLORS["gray"]]
    ax[0].bar(x, means, yerr=errs, capsize=3, color=colors, alpha=.9)
    for i, k in enumerate(sources):
        ax[0].scatter(np.full(3, i) + np.array([-.08, 0, .08]), vals[k], color=COLORS["navy"], s=12, zorder=3)
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("Scaled RMSE ↓")
    ax[0].set_title("One-step parameter substitution")

    horizon = np.arange(1, len(functional[0]["multi_horizon"]["probe"]) + 1)
    for k, label, color in zip(sources, labels, colors):
        curves = np.array([r["multi_horizon"][k] for r in functional])
        mu, sd = curves.mean(0), curves.std(0, ddof=1)
        ax[1].plot(horizon, mu, label=label, color=color, lw=2)
        ax[1].fill_between(horizon, mu - sd, mu + sd, color=color, alpha=.13)
    ax[1].set_xlabel("Rollout horizon")
    ax[1].set_ylabel("Scaled RMSE ↓")
    ax[1].set_title("Equation-based rollout")
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle("PokeWorld: functional evaluation of inferred physical variables (3 seeds)", weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "fig2_pokeworld_functional")

    # Prediction premise and label-based decodability.
    names = ["mass", "contact_stiffness", "drag"]
    pretty = ["Mass", "Stiffness", "Drag"]
    pred = np.array([[r["results"]["predictive/decodable"][n] for n in names] for r in decod])
    phys = np.array([[r["results"]["physwm/decodable"][n] for n in names] for r in decod])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    q_a = [r["prediction"]["predictive"]["path_a_query_vs_dataset"] for r in decod]
    persistence = [r["prediction"]["predictive"]["persistence_query_vs_dataset"] for r in decod]
    ax[0].bar([0, 1], [mean(q_a), mean(persistence)],
              yerr=[stdev(q_a), stdev(persistence)], capsize=3,
              color=[COLORS["blue"], COLORS["gray"]])
    ax[0].set_xticks([0, 1], ["Neural predictor", "Persistence"])
    ax[0].set_ylabel("Normalized RMSE ↓")
    ax[0].set_title("Predictive premise")
    width = .34
    x = np.arange(3)
    ax[1].bar(x - width/2, pred.mean(0), width, yerr=pred.std(0, ddof=1),
              capsize=3, label="Predictive", color=COLORS["gray"])
    ax[1].bar(x + width/2, phys.mean(0), width, yerr=phys.std(0, ddof=1),
              capsize=3, label="PhysWM", color=COLORS["cyan"])
    ax[1].axhline(0, color=COLORS["navy"], lw=.8)
    ax[1].set_xticks(x, pretty)
    ax[1].set_ylabel(r"Held-out $R^2$ ↑")
    ax[1].set_title("Supervised linear readout")
    ax[1].legend(frameon=False)
    fig.suptitle("PokeWorld: accurate prediction but mixed semantic parameter recovery", weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "fig3_pokeworld_prediction_decodability")

    rows = []
    for seed, (d, f) in enumerate(zip(decod, functional)):
        row = {
            "benchmark": "PokeWorld", "seed": seed,
            "path_a_query_rmse": d["prediction"]["predictive"]["path_a_query_vs_dataset"],
            "persistence_query_rmse": d["prediction"]["predictive"]["persistence_query_vs_dataset"],
            "r2_predictive_mass": d["results"]["predictive/decodable"]["mass"],
            "r2_physwm_mass": d["results"]["physwm/decodable"]["mass"],
            "r2_predictive_stiffness": d["results"]["predictive/decodable"]["contact_stiffness"],
            "r2_physwm_stiffness": d["results"]["physwm/decodable"]["contact_stiffness"],
            "r2_predictive_drag": d["results"]["predictive/decodable"]["drag"],
            "r2_physwm_drag": d["results"]["physwm/decodable"]["drag"],
            "sub_probe": f["substitution"]["probe"], "sub_true": f["substitution"]["true"],
            "sub_shuffled": f["substitution"]["shuffled"], "sub_nominal": f["substitution"]["nominal"],
            "h7_probe": f["multi_horizon"]["probe"][-1],
            "h7_true": f["multi_horizon"]["true"][-1],
            "h7_shuffled": f["multi_horizon"]["shuffled"][-1],
            "h7_nominal": f["multi_horizon"]["nominal"][-1],
        }
        rows.append(row)
    return rows


def pusht_figure(runs: Path, out: Path) -> list[dict]:
    data = [load(runs / "pusht-cartpole-matrix" / f"pusht_rand_xbench_seed{s}.json") for s in range(3)]
    sources, labels = ["probe", "nominal", "shuffled"], ["Inferred", "Nominal", "Shuffled"]
    colors = [COLORS["cyan"], COLORS["gray"], COLORS["red"]]
    values = {k: [r["substitution"][k] for r in data] for k in sources}
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    x = np.arange(3)
    ax[0].bar(x, [mean(values[k]) for k in sources],
              yerr=[stdev(values[k]) for k in sources], capsize=3, color=colors)
    for i, k in enumerate(sources):
        ax[0].scatter(np.full(3, i) + np.array([-.08, 0, .08]), values[k],
                      color=COLORS["navy"], s=12, zorder=3)
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("Scaled RMSE ↓")
    ax[0].set_title("Episode-specific substitution")
    path_a = [r["prediction"]["path_a"] for r in data]
    persistence = [r["prediction"]["persistence"] for r in data]
    width = .34
    seeds = np.arange(3)
    ax[1].bar(seeds-width/2, path_a, width, label="Neural predictor", color=COLORS["blue"])
    ax[1].bar(seeds+width/2, persistence, width, label="Persistence", color=COLORS["gray"])
    ax[1].set_xticks(seeds, ["Seed 0", "Seed 1", "Seed 2"])
    ax[1].set_ylabel("Scaled RMSE ↓")
    ax[1].set_title("Prediction premise is not uniform")
    ax[1].legend(frameon=False)
    fig.suptitle("PushT: inferred effective dynamics beat shuffled parameters", weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "fig4_pusht_effective_dynamics")
    return [{
        "benchmark": "PushT", "seed": s,
        "path_a_rmse": r["prediction"]["path_a"],
        "persistence_rmse": r["prediction"]["persistence"],
        "sub_probe": r["substitution"]["probe"],
        "sub_nominal": r["substitution"]["nominal"],
        "sub_shuffled": r["substitution"]["shuffled"],
        "beats_nominal": r["beats_nominal"], "beats_shuffled": r["beats_shuffled"],
    } for s, r in enumerate(data)]


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out: Path, poke: list[dict], pusht: list[dict]) -> None:
    def ms(rows, key):
        xs = [float(r[key]) for r in rows]
        return {"mean": mean(xs), "sample_sd": stdev(xs), "values": xs}

    summary = {
        "provenance": {
            "note": "Generated from explicitly selected JSON files; pre-fix Fetch excluded.",
            "pokeworld_decodability": "pilot-2048ep/A_pilot_2048ep_seed{0,1,2}.json",
            "pokeworld_functional": "functional-2048/func2048_seed{0,1,2}.json",
            "pusht_functional": "pusht-cartpole-matrix/pusht_rand_xbench_seed{0,1,2}.json",
        },
        "pokeworld": {k: ms(poke, k) for k in [
            "path_a_query_rmse", "persistence_query_rmse",
            "r2_predictive_mass", "r2_physwm_mass",
            "r2_predictive_stiffness", "r2_physwm_stiffness",
            "r2_predictive_drag", "r2_physwm_drag",
            "sub_probe", "sub_true", "sub_shuffled", "sub_nominal",
            "h7_probe", "h7_true", "h7_shuffled", "h7_nominal",
        ]},
        "pusht": {k: ms(pusht, k) for k in [
            "path_a_rmse", "persistence_rmse", "sub_probe", "sub_nominal", "sub_shuffled"
        ]},
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/physwm-artifacts/runs"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_style()
    architecture_figure(args.output)
    poke = pokeworld_figures(args.runs_root, args.output)
    pusht = pusht_figure(args.runs_root, args.output)
    write_csv(args.output / "pokeworld_metrics.csv", poke)
    write_csv(args.output / "pusht_metrics.csv", pusht)
    write_summary(args.output, poke, pusht)
    print(f"Wrote paper assets to {args.output}")


if __name__ == "__main__":
    main()
