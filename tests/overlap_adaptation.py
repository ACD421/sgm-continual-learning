#!/usr/bin/env python3
"""
SGM Adaptation and Evolution Test
=================================

This script demonstrates how the Sparse Gradient Memory (SGM) network adapts
to a sequence of related tasks while locking important blocks to preserve
prior knowledge. We compare SGM against a baseline network that trains
without any locking. Both networks are small multi-layer perceptrons trained
on a series of overlapping MNIST-like binary classification tasks generated
from the SyntheticMNIST class in sgm_experiments.py.

Tasks and setup:

  - Five tasks are defined: 0v1, 0v2, 0v3, 0v4, and 0v5. Each task is a
    binary classification problem distinguishing two MNIST digits. The first
    digit in the pair maps to label 0 and the second digit maps to label 1.
    Note that all tasks share digit 0, creating overlap that forces the
    network to adapt previously learned knowledge.

  - For each task, we generate 100 samples per class using SyntheticMNIST.
    The networks are trained via an evolutionary strategy for a few epochs.

  - After each task, we evaluate accuracy on every task seen so far and
    compute retention on the first task (0v1). The number of locked blocks
    in the SGM network is recorded to measure how much capacity has been
    consumed.

To run:

    python adapt_test.py

The script prints a summary table showing task accuracies for both the
SGM and baseline networks, retention on the first task after each new
task, and the number of locked blocks used by SGM.
"""

import numpy as np
from pathlib import Path

# Import SyntheticMNIST and SGMNetwork from the existing experiments module
from sgm_experiments import SyntheticMNIST, Config, SGMNetwork


class BaselineNetwork(SGMNetwork):
    """Baseline network that trains without locking any blocks."""

    def _lock_important_blocks(self, X: np.ndarray, y: np.ndarray, task_name: str) -> int:
        # Do not lock any blocks for the baseline.
        return 0


def run_adaptation_test():
    # Configuration: small hidden layer and few epochs for speed
    cfg = Config(input_dim=784, hidden_dim=64, output_dim=2,
                 block_size=64, lr=0.05, epochs_per_task=5,
                 n_samples_per_class=100, seed=42)

    # Initialize dataset generator
    mnist = SyntheticMNIST(cfg)

    # Define overlapping tasks
    tasks = [
        ("0v1", [0, 1]),
        ("0v2", [0, 2]),
        ("0v3", [0, 3]),
        ("0v4", [0, 4]),
        ("0v5", [0, 5]),
    ]

    # Pre-generate data for all tasks
    task_data = {}
    for name, digits in tasks:
        X, y = mnist.generate(digits, cfg.n_samples_per_class)
        # Binary labels: first digit -> 0, second digit -> 1
        y_bin = (y == digits[1]).astype(np.int32)
        task_data[name] = (X, y_bin)

    # Initialize SGM and baseline networks
    net_sgm = SGMNetwork(cfg)
    net_baseline = BaselineNetwork(cfg)

    # Record results
    results = []
    initial_acc_sgm = None
    initial_acc_base = None

    # Train each task sequentially
    for idx, (task_name, digits) in enumerate(tasks, start=1):
        X_train, y_train = task_data[task_name]

        # Train SGM network
        res_sgm = net_sgm.train_task(X_train, y_train, task_name,
                                     epochs=cfg.epochs_per_task, lr=cfg.lr)
        # Train baseline network
        res_base = net_baseline.train_task(X_train, y_train, task_name,
                                           epochs=cfg.epochs_per_task, lr=cfg.lr)

        # Evaluate on all tasks seen so far
        acc_sgm = {}
        acc_base = {}
        for eval_name, _ in tasks[:idx]:
            X_eval, y_eval = task_data[eval_name]
            acc_sgm[eval_name] = net_sgm.accuracy(X_eval, y_eval)
            acc_base[eval_name] = net_baseline.accuracy(X_eval, y_eval)

        # Record initial accuracy on the first task after training it
        if idx == 1:
            initial_acc_sgm = acc_sgm["0v1"]
            initial_acc_base = acc_base["0v1"]

        # Compute retention on first task (0v1)
        retention_sgm = acc_sgm["0v1"] / initial_acc_sgm if initial_acc_sgm else 0.0
        retention_base = acc_base["0v1"] / initial_acc_base if initial_acc_base else 0.0

        # Gather stats
        stats_sgm = net_sgm.stats()
        results.append({
            "task": task_name,
            "sgm_acc": acc_sgm.copy(),
            "baseline_acc": acc_base.copy(),
            "retention_sgm": retention_sgm,
            "retention_baseline": retention_base,
            "locked_blocks": stats_sgm["locked_blocks"],
            "locked_pct": stats_sgm["locked_pct"],
        })

    # Print summary table
    print("\n=== Adaptation Test Summary ===")
    header = (
        "Task   | SGM Accuracies                 | Baseline Accuracies           | "
        "Retention SGM | Retention Baseline | Locked Blocks (SGM)"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        # Format accuracy strings
        sgm_acc_str = ", ".join(f"{k}:{v*100:.1f}%" for k, v in r["sgm_acc"].items())
        base_acc_str = ", ".join(f"{k}:{v*100:.1f}%" for k, v in r["baseline_acc"].items())
        print(
            f"{r['task']:5} | "
            f"{sgm_acc_str:<30} | "
            f"{base_acc_str:<30} | "
            f"{r['retention_sgm']*100:>7.1f}%      | "
            f"{r['retention_baseline']*100:>7.1f}%         | "
            f"{r['locked_blocks']} ({r['locked_pct']:.1f}%)"
        )


if __name__ == "__main__":
    run_adaptation_test()