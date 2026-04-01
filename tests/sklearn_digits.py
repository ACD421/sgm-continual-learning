"""
Real-World Dataset Tests for Sparse Gradient Mutation (SGM)
=========================================================

This script evaluates the SGM coalition-locking primitive on a real
dataset from scikit-learn.  It uses the handwritten digits dataset
(64 features, 10 classes) to create sequential learning tasks.  Each
task masks a distinct subset of input features and trains a feed-forward
neural network to predict the digit labels.  The tasks are designed
to overlap only in the dataset (labels) but use different input
dimensions, making them realistic yet challenging.

We compare a baseline (no locking) implementation of SGM with a
coalition-locking variant.  For each variant we measure the retention
ratio on earlier tasks when new tasks are learned, and we report how
many parameters are locked after each task.  Lower retention ratios
indicate less forgetting (1.0 = perfect retention).  We use mean
squared error between the network output and one-hot label vectors as
the loss function.

Usage: run this file directly with ``python3 sgm_real_world_tests.py``.

Note: These experiments use a reduced evaluation budget and small
subsets of the dataset to fit within time constraints.  They serve
as a proof of concept; for more robust results increase ``n_evals``
and consider using a larger sample of the dataset.
"""

import numpy as np
from sklearn.datasets import load_digits
from typing import List, Tuple

from sgm_model_tests import NNModel, SGMBaselineModel, SGMWithLockingModel


class DatasetModelTask:
    """Task using a real dataset with masked inputs and one-hot targets."""

    def __init__(self, model: NNModel, input_mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Args:
            model: A neural network model (NNModel) whose ``input_dim`` matches ``X.shape[1]``.
            input_mask: Boolean array of shape (input_dim,) indicating which input features are active.
            X: Input samples of shape (n_samples, input_dim).
            Y: Target labels as one-hot vectors of shape (n_samples, output_dim).
        """
        self.model = model
        self.input_mask = input_mask.astype(bool)
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        assert X.shape[1] == model.input_dim
        assert Y.shape[1] == model.output_dim

    def loss(self, params: np.ndarray) -> float:
        total = 0.0
        for x, y in zip(self.X, self.Y):
            # apply input mask (zero out inactive dims)
            x_masked = x * self.input_mask.astype(np.float32)
            # forward pass
            out = self.model.forward_given_params(params, x_masked)
            total += np.mean((out - y) ** 2)
        return total / len(self.X)


def build_dataset_tasks(model: NNModel, n_tasks: int, sample_size: int = 200, seed: int = 0) -> List[DatasetModelTask]:
    """Create tasks from the digits dataset.

    Each task masks a distinct contiguous block of input features.  A
    fixed subset of the dataset is used for all tasks.  Targets are
    one-hot encodings of the digit labels.

    Args:
        model: Neural network model (NNModel) with input dimension 64 and output dimension 10.
        n_tasks: Number of tasks to create.
        sample_size: Number of samples to use from the dataset (randomly selected).
        seed: RNG seed for reproducibility.

    Returns:
        List of DatasetModelTask objects.
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    # normalize inputs to [0, 1]
    X = X / 16.0
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=sample_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    # one-hot encode targets
    Y_subset = np.zeros((sample_size, model.output_dim), dtype=np.float32)
    for i, label in enumerate(y_subset):
        Y_subset[i, label] = 1.0
    # build masks
    input_dim = model.input_dim
    segment = input_dim // n_tasks
    tasks: List[DatasetModelTask] = []
    for i in range(n_tasks):
        mask = np.zeros(input_dim, dtype=bool)
        start = i * segment
        end = (i + 1) * segment if i < n_tasks - 1 else input_dim
        mask[start:end] = True
        tasks.append(DatasetModelTask(model, mask, X_subset, Y_subset))
    return tasks


def run_real_world_scenario(n_tasks: int = 3, n_evals: int = 200, n_runs: int = 1) -> Tuple[List[float], List[float], List[List[float]]]:
    """Run a real-world dataset scenario with baseline and locking SGM.

    Creates a feed-forward NN model (64-32-16-10) and sequentially trains
    it on ``n_tasks`` masked tasks derived from the digits dataset.  For
    each run we record the retention ratios on tasks 0 through ``n_tasks-2``
    and the fraction of locked parameters after each task.

    Args:
        n_tasks: Number of sequential tasks.
        n_evals: Evaluation budget per task.
        n_runs: Number of independent runs.

    Returns:
        Tuple (baseline_ret_means, locking_ret_means, locking_locked_fracs)
        where:
            - baseline_ret_means is a list of mean retention ratios across runs for the baseline.
            - locking_ret_means is a list of mean retention ratios across runs for the locked variant.
            - locking_locked_fracs is a nested list of locked fraction sequences per run.
    """
    baseline_ret_all = []
    locking_ret_all = []
    locked_fracs_runs: List[List[float]] = []
    for run in range(n_runs):
        np.random.seed(run * 1007)
        # build model: 64 inputs, two hidden layers and 10 outputs
        model = NNModel(input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=10)
        tasks = build_dataset_tasks(model, n_tasks, sample_size=200, seed=run * 37)
        # baseline
        base_model = SGMBaselineModel(model.total_params)
        during_losses = []
        for task in tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            during_losses.append(base_model.best_loss)
        final_losses = [task.loss(base_model.best_params) for task in tasks]
        ratios = [final_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        baseline_ret_all.append(ratios)
        # locking
        lock_model = SGMWithLockingModel(model.total_params)
        during_losses_l = []
        locked_fracs = []
        for task in tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            during_losses_l.append(lock_model.best_loss)
            locked_fracs.append(float(np.mean(lock_model.lock)))
        final_losses_l = [task.loss(lock_model.best_params) for task in tasks]
        ratios_l = [final_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        locking_ret_all.append(ratios_l)
        locked_fracs_runs.append(locked_fracs)
    # compute means across runs
    baseline_ret_means = list(np.mean(baseline_ret_all, axis=0))
    locking_ret_means = list(np.mean(locking_ret_all, axis=0))
    return baseline_ret_means, locking_ret_means, locked_fracs_runs


def main():
    # Run a quick real-world scenario
    n_tasks = 3
    n_runs = 2
    n_evals = 200
    baseline_ret, locking_ret, locked_fracs_runs = run_real_world_scenario(
        n_tasks=n_tasks, n_evals=n_evals, n_runs=n_runs
    )
    print("\nReal-World Dataset Scenario (digits) with 64-32-16-10 NN")
    print(f"Baseline retention ratios per task: {[f'{r:.2f}' for r in baseline_ret]}")
    print(f"Locked retention ratios per task:   {[f'{r:.2f}' for r in locking_ret]}")
    # Compute mean locked fraction per task across runs
    locked_fracs_mean = list(np.mean(locked_fracs_runs, axis=0))
    for i, frac in enumerate(locked_fracs_mean, start=1):
        print(f"  After task {i}: fraction of locked params ~ {frac:.2f}")


if __name__ == "__main__":
    main()