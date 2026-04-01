"""
SGM CIFAR/Natural Language Simulation
=====================================

This script simulates hierarchical image tasks and natural language tasks
using the Sparse Gradient Mutation (SGM) framework.  The goal is to
approximate more realistic scenarios, such as CIFAR-like overlapping
feature spaces and bag-of-words text classification, under the
constraints of the available environment (no external deep learning
libraries or large datasets).  We re-use the simple feed-forward model
and SGM algorithms from ``sgm_model_tests.py`` and define synthetic
tasks with overlapping input masks to mimic hierarchical structure and
shared vocabulary.

The script runs both a baseline evolutionary optimizer (no locking) and
the locked SGM on two sets of tasks:

1. **Hierarchical tasks:** Three tasks with partially overlapping
   regions of a 1-dimensional feature space.  These tasks simulate
   hierarchical image features, where low-level features are shared
   across tasks and higher-level features are unique.  Each task
   consists of random inputs masked by the active region and a random
   target output vector.

2. **Natural language tasks:** Three tasks where each task activates a
   different set of dimensions representing "words" in a
   bag-of-words encoding.  Overlaps between tasks simulate shared
   vocabulary.  Although no real text is used, the structure mimics
   the way classification tasks may share common terms.

For each set of tasks the script reports:
   * The retention ratio (final loss divided by best training loss) for
     both baseline and locked models.  A ratio near 1.0 indicates
     perfect memory of earlier tasks; higher values indicate forgetting.
   * The fraction of parameters locked after each task for the locked
     model.

Run this script with ``python sgm_cifar_natty_simulation.py``.  It
prints results to stdout.  This file can be adapted to explore
additional scenarios.
"""

import numpy as np
from typing import List, Tuple

from sgm_model_tests import NNModel, ModelTask, SGMBaselineModel, SGMWithLockingModel


def run_task_sequence(tasks: List[ModelTask], model_dim: int, n_evals: int = 300) -> Tuple[float, List[float]]:
    """
    Train baseline and locked SGM on a sequence of tasks and compute
    retention ratio and locked fraction per task.

    Args:
        tasks: List of ModelTask instances.
        model_dim: Dimension of the flattened model parameter vector.
        n_evals: Number of fitness evaluations per task.

    Returns:
        (baseline_retention, locked_retention), (baseline_locked_fractions, locked_fractions)
    """
    # baseline
    baseline = SGMBaselineModel(model_dim)
    baseline_losses = []
    for task in tasks:
        baseline.reset()
        loss_before = baseline.step(task, n_evals)
        baseline_losses.append(loss_before)
    baseline_final_losses = [task.loss(baseline.best_params) for task in tasks]
    baseline_ret = np.mean([
        baseline_final_losses[i] / baseline_losses[i] if baseline_losses[i] > 0 else 1.0
        for i in range(len(tasks) - 1)
    ])

    # locked
    locked = SGMWithLockingModel(model_dim)
    locked_losses = []
    locked_fractions = []
    for task in tasks:
        locked.reset()
        loss_before = locked.step(task, n_evals)
        locked_losses.append(loss_before)
        locked_fractions.append(float(np.mean(locked.lock)))
    locked_final_losses = [task.loss(locked.best_params) for task in tasks]
    locked_ret = np.mean([
        locked_final_losses[i] / locked_losses[i] if locked_losses[i] > 0 else 1.0
        for i in range(len(tasks) - 1)
    ])
    return (baseline_ret, locked_ret), locked_fractions


def simulate_hierarchical_tasks() -> None:
    """Simulate a set of hierarchical tasks and print retention results."""
    print("\n=== Hierarchical CIFAR-like tasks ===")
    # Use a moderate input dimension to approximate image features
    input_dim = 120
    model = NNModel(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=16)
    # Define three tasks with overlapping active regions
    tasks = []
    # Task1: dims 0--49 active (base + unique1)
    mask1 = np.zeros(input_dim, dtype=bool)
    mask1[0:50] = True
    target1 = np.random.randn(model.output_dim).astype(np.float32) * 0.5
    tasks.append(ModelTask(model, mask1, target1, n_samples=3, seed=42))
    # Task2: dims 30--79 active (overlap with task1 on 30--49)
    mask2 = np.zeros(input_dim, dtype=bool)
    mask2[30:80] = True
    target2 = np.random.randn(model.output_dim).astype(np.float32) * 0.5
    tasks.append(ModelTask(model, mask2, target2, n_samples=3, seed=43))
    # Task3: dims 60--109 active (overlap with task2 on 60--79)
    mask3 = np.zeros(input_dim, dtype=bool)
    mask3[60:110] = True
    target3 = np.random.randn(model.output_dim).astype(np.float32) * 0.5
    tasks.append(ModelTask(model, mask3, target3, n_samples=3, seed=44))
    # Run sequence
    (base_ret, lock_ret), lock_fracs = run_task_sequence(tasks, model.total_params, n_evals=200)
    print(f"Retention (baseline): {base_ret:.3f}, (locked): {lock_ret:.3f}")
    print("Locked fraction per task:", [f"{f*100:.1f}%" for f in lock_fracs])


def simulate_natural_language_tasks() -> None:
    """Simulate a set of bag-of-words natural language tasks and print results."""
    print("\n=== Natural language bag-of-words tasks ===")
    # Use a smaller dimension representing vocabulary size
    vocab_dim = 100
    model = NNModel(input_dim=vocab_dim, hidden_dim1=64, hidden_dim2=32, output_dim=8)
    tasks = []
    # Task1: 'sports' vs 'politics': dims 0--39 active (0--9 common words, 10--19 sports, 20--29 politics, 30--39 unique overlap)
    mask1 = np.zeros(vocab_dim, dtype=bool)
    mask1[0:40] = True
    target1 = np.random.randn(model.output_dim).astype(np.float32) * 0.5
    tasks.append(ModelTask(model, mask1, target1, n_samples=3, seed=100))
    # Task2: 'politics' vs 'science': dims 20--59 active (20--29 politics again, 30--39 overlap, 40--49 science, 50--59 unique overlap)
    mask2 = np.zeros(vocab_dim, dtype=bool)
    mask2[20:60] = True
    target2 = np.random.randn(model.output_dim).astype(np.float32) * 0.5
    tasks.append(ModelTask(model, mask2, target2, n_samples=3, seed=101))
    # Task3: 'science' vs 'arts': dims 40--79 active (40--49 science again, 50--59 overlap, 60--69 arts, 70--79 unique overlap)
    mask3 = np.zeros(vocab_dim, dtype=bool)
    mask3[40:80] = True
    target3 = np.random.randn(model.output_dim).astype(np.float32) * 0.5
    tasks.append(ModelTask(model, mask3, target3, n_samples=3, seed=102))
    # Run sequence
    (base_ret, lock_ret), lock_fracs = run_task_sequence(tasks, model.total_params, n_evals=200)
    print(f"Retention (baseline): {base_ret:.3f}, (locked): {lock_ret:.3f}")
    print("Locked fraction per task:", [f"{f*100:.1f}%" for f in lock_fracs])


def main() -> None:
    simulate_hierarchical_tasks()
    simulate_natural_language_tasks()


if __name__ == "__main__":
    main()