#!/usr/bin/env python3
"""
Standalone parallel experiment runner for TP2 (Redes Neurais Artificiais - DCC642 UFMG).

Uses `learn.py` directly without modifying it, runs all required hyperparameter sweeps
and the best (>= 90% test accuracy) configuration in parallel without race conditions,
saves isolated outputs per experiment, generates consolidated JSON/CSV summaries and
publication-ready matplotlib figures, and deploys the best model weights to `src/model/`.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

# Avoid BLAS thread oversubscription when running multiple worker processes in parallel.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import learn  # noqa: E402

EXPERIMENTS_DIR = SRC_DIR / "experiments"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"
MODEL_DIR = SRC_DIR / "model"

SEED = 42
INPUT_SIZE = 784
OUTPUT_SIZE = 10

# Baseline configuration shared across all three sweeps
BASELINE = {
    "learning_rate": 0.1,
    "epochs": 200,
    "hidden_size": 64,
}

# Shared read-only dataset populated in parent before fork()
_SHARED_DATA = {}


def _save_experiment_artifacts(
    exp_dir: Path,
    exp_id: str,
    sweep: str,
    learning_rate: float,
    epochs: int,
    hidden_size: int,
    parameters,
    losses: list[float],
    accuracies: list[float],
    elapsed_seconds: float,
) -> dict:
    """Save weights, CSV curves, and metrics.json into an isolated experiment directory."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    W1, b1, W2, b2 = parameters

    np.save(exp_dir / "W1.npy", W1)
    np.save(exp_dir / "b1.npy", b1)
    np.save(exp_dir / "W2.npy", W2)
    np.save(exp_dir / "b2.npy", b2)

    np.savetxt(exp_dir / "losses.csv", np.array(losses, dtype=np.float64), delimiter=",")
    np.savetxt(exp_dir / "accuracies.csv", np.array(accuracies, dtype=np.float64), delimiter=",")

    # Every-100-epochs curve matching server.py / plot.py convention (epochs 100, 200, ...)
    indices_100 = [i for i in range(len(losses)) if (i + 1) % 100 == 0]
    if not indices_100:
        indices_100 = [len(losses) - 1]
    losses_100 = [losses[i] for i in indices_100]
    accs_100 = [accuracies[i] for i in indices_100]
    np.savetxt(exp_dir / "losses_every100.csv", np.array(losses_100, dtype=np.float64), delimiter=",")
    np.savetxt(exp_dir / "accuracies_every100.csv", np.array(accs_100, dtype=np.float64), delimiter=",")

    best_idx = int(np.argmax(accuracies))
    metrics = {
        "exp_id": exp_id,
        "sweep": sweep,
        "config": {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "seed": SEED,
            "input_size": INPUT_SIZE,
            "output_size": OUTPUT_SIZE,
        },
        "final_train_loss": float(losses[-1]),
        "final_test_acc": float(accuracies[-1]),
        "best_test_acc": float(accuracies[best_idx]),
        "best_test_acc_epoch": best_idx + 1,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "losses_history": [float(v) for v in losses],
        "accuracies_history": [float(v) for v in accuracies],
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _run_job(job: dict) -> list[dict]:
    """
    Execute one training trajectory in a worker process.
    Supports optional `checkpoints` (dict mapping epoch -> (exp_id, sweep)) so that
    deterministic prefix runs (e.g., epochs=50, 100, 200, 500 with identical seed/lr/hidden)
    save exact weights, curves, and cumulative elapsed times without redundant compute.
    """
    exp_id = job["exp_id"]
    sweep = job["sweep"]
    lr = float(job["learning_rate"])
    max_epochs = int(job["epochs"])
    hidden_size = int(job["hidden_size"])
    checkpoints = job.get("checkpoints")  # {epoch_num: (ckpt_exp_id, ckpt_sweep)}
    blas_threads = int(job.get("blas_threads", 1))

    X_train = _SHARED_DATA["X_train"]
    X_test = _SHARED_DATA["X_test"]
    y_train = _SHARED_DATA["y_train"]
    y_test = _SHARED_DATA["y_test"]

    results = []
    with threadpool_limits(limits=blas_threads):
        np.random.seed(SEED)
        parameters = learn.initialize_parameters(INPUT_SIZE, hidden_size, OUTPUT_SIZE)

        losses: list[float] = []
        accuracies: list[float] = []

        t0 = time.perf_counter()
        for ep in range(1, max_epochs + 1):
            parameters, loss = learn.gradient_descent_step(X_train, y_train, parameters, learning_rate=lr)
            y_pred_test = learn.forward(X_test, parameters)[-1]
            acc = float(learn.accuracy(y_test, y_pred_test))

            losses.append(float(loss))
            accuracies.append(acc)

            if ep == 1 or ep % 50 == 0 or ep == max_epochs:
                elapsed = time.perf_counter() - t0
                print(
                    f"[{exp_id}] epoch {ep:4d}/{max_epochs} | "
                    f"loss={loss:.4f} | test_acc={acc*100:.2f}% | elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if checkpoints and ep in checkpoints:
                ckpt_id, ckpt_sweep = checkpoints[ep]
                elapsed = time.perf_counter() - t0
                m = _save_experiment_artifacts(
                    EXPERIMENTS_DIR / ckpt_id,
                    ckpt_id,
                    ckpt_sweep,
                    lr,
                    ep,
                    hidden_size,
                    parameters,
                    losses,
                    accuracies,
                    elapsed,
                )
                results.append(m)

        if not checkpoints:
            elapsed = time.perf_counter() - t0
            m = _save_experiment_artifacts(
                EXPERIMENTS_DIR / exp_id,
                exp_id,
                sweep,
                lr,
                max_epochs,
                hidden_size,
                parameters,
                losses,
                accuracies,
                elapsed,
            )
            results.append(m)

    return results


def build_sweep_jobs() -> list[dict]:
    """
    Build all jobs for the 3 required sweeps from TP2 Section 4:
      - Sweep 1 (learning_rate): 0.001, 0.01, 0.1, 0.5  (epochs=200, hidden=64)
      - Sweep 2 (epochs):        50, 100, 200, 500      (lr=0.1, hidden=64)
      - Sweep 3 (hidden_size):   4, 16, 64, 256         (lr=0.1, epochs=200)

    Because full-batch gradient descent with fixed seed=42 is strictly deterministic,
    Sweep 2 checkpoints at epochs 50, 100, 200 (baseline), and 500 produce the exact
    same weights, losses, accuracies, and cumulative wall-clock times as separate runs.
    """
    jobs = [
        # Longest job first for optimal parallel scheduling across 4 workers
        {
            "exp_id": "sweep3_hidden_256",
            "sweep": "hidden_size",
            "learning_rate": BASELINE["learning_rate"],
            "epochs": BASELINE["epochs"],
            "hidden_size": 256,
        },
        {
            "exp_id": "sweep2_epochs_500",
            "sweep": "epochs",
            "learning_rate": BASELINE["learning_rate"],
            "epochs": 500,
            "hidden_size": BASELINE["hidden_size"],
            "checkpoints": {
                50: ("sweep2_epochs_50", "epochs"),
                100: ("sweep2_epochs_100", "epochs"),
                200: ("baseline_lr0.1_ep200_h64", "baseline"),
                500: ("sweep2_epochs_500", "epochs"),
            },
        },
        {
            "exp_id": "sweep1_lr_0.001",
            "sweep": "learning_rate",
            "learning_rate": 0.001,
            "epochs": BASELINE["epochs"],
            "hidden_size": BASELINE["hidden_size"],
        },
        {
            "exp_id": "sweep1_lr_0.01",
            "sweep": "learning_rate",
            "learning_rate": 0.01,
            "epochs": BASELINE["epochs"],
            "hidden_size": BASELINE["hidden_size"],
        },
        {
            "exp_id": "sweep1_lr_0.5",
            "sweep": "learning_rate",
            "learning_rate": 0.5,
            "epochs": BASELINE["epochs"],
            "hidden_size": BASELINE["hidden_size"],
        },
        {
            "exp_id": "sweep3_hidden_16",
            "sweep": "hidden_size",
            "learning_rate": BASELINE["learning_rate"],
            "epochs": BASELINE["epochs"],
            "hidden_size": 16,
        },
        {
            "exp_id": "sweep3_hidden_4",
            "sweep": "hidden_size",
            "learning_rate": BASELINE["learning_rate"],
            "epochs": BASELINE["epochs"],
            "hidden_size": 4,
        },
    ]
    return jobs


def build_best_job(lr: float, epochs: int, hidden_size: int, blas_threads: int) -> dict:
    return {
        "exp_id": "best_model",
        "sweep": "best",
        "learning_rate": lr,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "blas_threads": blas_threads,
    }


def load_all_metrics() -> dict[str, dict]:
    """Load metrics.json from every experiment subdirectory."""
    all_metrics = {}
    if not EXPERIMENTS_DIR.exists():
        return all_metrics
    for p in sorted(EXPERIMENTS_DIR.glob("*/metrics.json")):
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
            all_metrics[m["exp_id"]] = m
    return all_metrics


def write_consolidated_summary(all_metrics: dict[str, dict]) -> None:
    """Write experiments/summary.json and experiments/summary.csv cleanly formatted."""
    ordered_rows = []

    def add_row(sweep_name: str, exp_id: str):
        if exp_id not in all_metrics:
            return
        m = all_metrics[exp_id]
        cfg = m["config"]
        ordered_rows.append(
            {
                "sweep": sweep_name,
                "exp_id": exp_id,
                "learning_rate": cfg["learning_rate"],
                "epochs": cfg["epochs"],
                "hidden_size": cfg["hidden_size"],
                "final_train_loss": round(m["final_train_loss"], 6),
                "final_test_acc": round(m["final_test_acc"], 6),
                "final_test_acc_pct": round(m["final_test_acc"] * 100.0, 2),
                "best_test_acc": round(m["best_test_acc"], 6),
                "best_test_acc_pct": round(m["best_test_acc"] * 100.0, 2),
                "best_test_acc_epoch": m["best_test_acc_epoch"],
                "elapsed_seconds": m["elapsed_seconds"],
            }
        )

    # Sweep 1: Learning Rate
    for eid in [
        "sweep1_lr_0.001",
        "sweep1_lr_0.01",
        "baseline_lr0.1_ep200_h64",
        "sweep1_lr_0.5",
    ]:
        add_row("learning_rate", eid)

    # Sweep 2: Epochs
    for eid in [
        "sweep2_epochs_50",
        "sweep2_epochs_100",
        "baseline_lr0.1_ep200_h64",
        "sweep2_epochs_500",
    ]:
        add_row("epochs", eid)

    # Sweep 3: Hidden Size
    for eid in [
        "sweep3_hidden_4",
        "sweep3_hidden_16",
        "baseline_lr0.1_ep200_h64",
        "sweep3_hidden_256",
    ]:
        add_row("hidden_size", eid)

    # Best Model
    add_row("best", "best_model")

    summary_json = {
        "baseline": BASELINE,
        "seed": SEED,
        "results": ordered_rows,
    }
    with open(EXPERIMENTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    if ordered_rows:
        fieldnames = list(ordered_rows[0].keys())
        with open(EXPERIMENTS_DIR / "summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_rows)

    print("\n=== Consolidated Summary ===")
    header = f"{'Sweep':<14} {'Exp ID':<26} {'LR':<6} {'Epochs':<6} {'Hidden':<6} {'Loss':<9} {'Test Acc':<9} {'Time(s)':<8}"
    print(header)
    print("-" * len(header))
    for r in ordered_rows:
        print(
            f"{r['sweep']:<14} {r['exp_id']:<26} {r['learning_rate']:<6} {r['epochs']:<6} "
            f"{r['hidden_size']:<6} {r['final_train_loss']:<9.4f} {r['final_test_acc_pct']:<8.2f}% {r['elapsed_seconds']:<8.1f}"
        )


def deploy_best_model() -> None:
    """Copy best_model weights and every-100-epochs CSVs to src/model/ for server.py & plot.py."""
    best_dir = EXPERIMENTS_DIR / "best_model"
    if not (best_dir / "W1.npy").exists():
        print("[WARN] best_model artifacts not found; skipping deploy to src/model/.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["W1.npy", "b1.npy", "W2.npy", "b2.npy"]:
        shutil.copy2(best_dir / fname, MODEL_DIR / fname)

    shutil.copy2(best_dir / "losses_every100.csv", MODEL_DIR / "losses.csv")
    shutil.copy2(best_dir / "accuracies_every100.csv", MODEL_DIR / "accuracies.csv")
    print(f"Deployed best_model weights and CSVs to {MODEL_DIR}")


def generate_plots(all_metrics: dict[str, dict]) -> None:
    """Generate publication-ready PNG (300 DPI) and PDF plots in src/experiments/plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "lines.linewidth": 2.0,
        }
    )

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    def save_fig(fig, stem: str):
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 1. Sweep 1: Learning Rate
    lr_items = [
        ("sweep1_lr_0.001", "lr = 0.001"),
        ("sweep1_lr_0.01", "lr = 0.01"),
        ("baseline_lr0.1_ep200_h64", "lr = 0.1 (baseline)"),
        ("sweep1_lr_0.5", "lr = 0.5"),
    ]
    if all(eid in all_metrics for eid, _ in lr_items):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for (eid, label), col in zip(lr_items, colors):
            m = all_metrics[eid]
            ep = np.arange(1, len(m["losses_history"]) + 1)
            ax1.plot(ep, m["losses_history"], label=label, color=col)
            ax2.plot(ep, [a * 100.0 for a in m["accuracies_history"]], label=label, color=col)
        ax1.set_title("Perda de Treinamento vs. Taxa de Aprendizado")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Perda (Entropia Cruzada)")
        ax1.legend()
        ax2.set_title("Acurácia no Teste vs. Taxa de Aprendizado")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Acurácia no Teste (%)")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="lower right")
        save_fig(fig, "sweep_learning_rate")

    # 2. Sweep 2: Epochs
    if "sweep2_epochs_500" in all_metrics:
        m = all_metrics["sweep2_epochs_500"]
        ep = np.arange(1, len(m["losses_history"]) + 1)
        accs_pct = [a * 100.0 for a in m["accuracies_history"]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

        ax1.plot(ep, m["losses_history"], color="#c0392b", label="Perda (lr=0.1, h=64)")
        ax2.plot(ep, accs_pct, color="#2980b9", label="Acurácia (lr=0.1, h=64)")

        milestones = [50, 100, 200, 500]
        for ms in milestones:
            loss_v = m["losses_history"][ms - 1]
            acc_v = accs_pct[ms - 1]
            ax1.axvline(ms, color="gray", linestyle=":", alpha=0.7)
            ax1.scatter([ms], [loss_v], color="#c0392b", zorder=5)
            ax1.annotate(
                f"{ms} ep\n({loss_v:.2f})",
                xy=(ms, loss_v),
                xytext=(-42 if ms == 500 else 5, 10 if ms == 500 else 6),
                textcoords="offset points",
                fontsize=8,
            )
            ax2.axvline(ms, color="gray", linestyle=":", alpha=0.7)
            ax2.scatter([ms], [acc_v], color="#2980b9", zorder=5)
            ax2.annotate(
                f"{ms} ep\n({acc_v:.1f}%)",
                xy=(ms, acc_v),
                xytext=(-36 if ms == 500 else 6, -24 if ms >= 100 else -18),
                textcoords="offset points",
                fontsize=8,
            )

        ax1.set_title("Perda de Treinamento ao Longo das Épocas")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Perda (Entropia Cruzada)")
        ax1.legend()
        ax2.set_title("Acurácia no Teste ao Longo das Épocas")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Acurácia no Teste (%)")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="lower right")
        save_fig(fig, "sweep_epochs")

    # 3. Sweep 3: Hidden Size
    h_items = [
        ("sweep3_hidden_4", "4 neurônios"),
        ("sweep3_hidden_16", "16 neurônios"),
        ("baseline_lr0.1_ep200_h64", "64 neurônios (baseline)"),
        ("sweep3_hidden_256", "256 neurônios"),
    ]
    if all(eid in all_metrics for eid, _ in h_items):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for (eid, label), col in zip(h_items, colors):
            m = all_metrics[eid]
            ep = np.arange(1, len(m["losses_history"]) + 1)
            ax1.plot(ep, m["losses_history"], label=label, color=col)
            ax2.plot(ep, [a * 100.0 for a in m["accuracies_history"]], label=label, color=col)
        ax1.set_title("Perda de Treinamento vs. Camada Oculta")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Perda (Entropia Cruzada)")
        ax1.legend()
        ax2.set_title("Acurácia no Teste vs. Camada Oculta")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Acurácia no Teste (%)")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="lower right")
        save_fig(fig, "sweep_hidden_size")

    # 4. Best Model Curves
    if "best_model" in all_metrics:
        m = all_metrics["best_model"]
        cfg = m["config"]
        ep = np.arange(1, len(m["losses_history"]) + 1)
        accs_pct = [a * 100.0 for a in m["accuracies_history"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        ax1.plot(ep, m["losses_history"], color="#c0392b", label="Perda de Treinamento")
        ax1.set_title(
            f"Melhor Modelo — Perda (lr={cfg['learning_rate']}, h={cfg['hidden_size']}, {cfg['epochs']} ep)"
        )
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Perda (Entropia Cruzada)")
        ax1.annotate(
            f"Final: {m['final_train_loss']:.4f}",
            xy=(ep[-1], m["final_train_loss"]),
            xytext=(-75, 18),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#c0392b"},
            fontsize=9,
            color="#c0392b",
        )
        ax1.legend()

        ax2.plot(ep, accs_pct, color="#1f77b4", label="Acurácia no Teste")
        ax2.axhline(90.0, color="#27ae60", linestyle="--", linewidth=1.3, label="Meta mínima (90%)")
        ax2.set_title(
            f"Melhor Modelo — Acurácia (Final: {m['final_test_acc']*100:.2f}%)"
        )
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Acurácia no Teste (%)")
        ax2.set_ylim(0, 100)
        ax2.annotate(
            f"Final: {m['final_test_acc']*100:.2f}%",
            xy=(ep[-1], accs_pct[-1]),
            xytext=(-85, -28),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
            fontsize=9,
            color="#1f77b4",
        )
        ax2.legend(loc="lower right")
        save_fig(fig, "best_model_curves")

    print(f"Saved plots to {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="TP2 Neural Network Experiment Runner")
    parser.add_argument(
        "--phase",
        choices=["sweeps", "best", "all", "plots"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel worker processes for sweeps",
    )
    parser.add_argument("--best-lr", type=float, default=0.5)
    parser.add_argument("--best-epochs", type=int, default=500)
    parser.add_argument("--best-hidden", type=int, default=256)
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "plots":
        all_metrics = load_all_metrics()
        write_consolidated_summary(all_metrics)
        generate_plots(all_metrics)
        return

    print(f"Loading MNIST once via learn.load_mnist() (CPUs={os.cpu_count()}, workers={args.workers})...", flush=True)
    t_load = time.perf_counter()
    X_train, X_test, y_train, y_test = learn.load_mnist()
    print(
        f"MNIST loaded in {time.perf_counter() - t_load:.2f}s | "
        f"X_train={X_train.shape}, X_test={X_test.shape}",
        flush=True,
    )
    _SHARED_DATA["X_train"] = X_train
    _SHARED_DATA["X_test"] = X_test
    _SHARED_DATA["y_train"] = y_train
    _SHARED_DATA["y_test"] = y_test

    if args.phase in ("sweeps", "all"):
        jobs = build_sweep_jobs()
        t_sweeps = time.perf_counter()
        print(f"\nLaunching {len(jobs)} sweep jobs across {args.workers} forked workers...", flush=True)
        ctx = get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_run_job, job): job["exp_id"] for job in jobs}
            for fut in as_completed(futures):
                jid = futures[fut]
                res_list = fut.result()
                for m in res_list:
                    print(
                        f"DONE [{m['exp_id']}] loss={m['final_train_loss']:.4f}, "
                        f"test_acc={m['final_test_acc']*100:.2f}%, time={m['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
        print(f"All sweeps completed in {time.perf_counter() - t_sweeps:.1f}s.", flush=True)

    if args.phase in ("best", "all"):
        # Use all CPU threads for the single best-model run
        n_threads = max(1, os.cpu_count() or 1)
        best_job = build_best_job(args.best_lr, args.best_epochs, args.best_hidden, blas_threads=n_threads)
        print(
            f"\nRunning best_model (lr={args.best_lr}, epochs={args.best_epochs}, "
            f"hidden={args.best_hidden}, blas_threads={n_threads})...",
            flush=True,
        )
        res_list = _run_job(best_job)
        m = res_list[0]
        print(
            f"DONE [best_model] loss={m['final_train_loss']:.4f}, "
            f"test_acc={m['final_test_acc']*100:.2f}%, time={m['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if m["final_test_acc"] < 0.90:
            print(
                f"[WARN] best_model test accuracy {m['final_test_acc']*100:.2f}% is below 90%!",
                flush=True,
            )
        deploy_best_model()

    all_metrics = load_all_metrics()
    write_consolidated_summary(all_metrics)
    generate_plots(all_metrics)


if __name__ == "__main__":
    main()
