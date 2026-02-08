"""
Dense Digit Tasks Experiment for Sparse Gradient Mutation (SGM)
================================================================

This script evaluates the SGM coalition‑locking primitive on a dense,
real dataset scenario.  We use the scikit‑learn handwritten digits dataset
and create a sequence of binary classification tasks, each involving two
distinct digits.  All tasks use the full set of input features (64 pixels),
so parameter interference is maximized: every task depends on every input
dimension.  This setting is intentionally challenging for SGM, as it
contrasts with previous experiments that masked non‑overlapping regions.

For each task, we build a small feed‑forward neural network with two hidden
layers and a binary output.  We train the network using the SGM baseline
(no locking) and the locked variant.  After each task is learned, we
evaluate retention on previously learned tasks and record the fraction of
parameters that were locked.  We expect the locked model to freeze
increasing portions of the parameter vector while still maintaining higher
retention compared to the baseline.

Usage: run this file directly with ``python3 dense_digit_tasks_experiment.py``.
"""

import numpy as np
from sklearn.datasets import load_digits
from typing import List, Tuple

from sgm_model_tests import NNModel, SGMBaselineModel, SGMWithLockingModel


class BinaryDigitTask:
    """Binary classification task for two digits using full input features."""

    def __init__(self, model: NNModel, digit_a: int, digit_b: int, X: np.ndarray, y: np.ndarray, max_samples: int = None) -> None:
        """
        Args:
            model: NNModel with input_dim matching number of features and output_dim equal to 2.
            digit_a: First digit label for class 0.
            digit_b: Second digit label for class 1.
            X: Input features, shape (n_samples, input_dim).
            y: Digit labels, shape (n_samples,).
        """
        assert model.output_dim == 2, "Model output_dim must be 2 for binary tasks."
        self.model = model
        self.input_dim = model.input_dim
        self.digit_a = digit_a
        self.digit_b = digit_b
        # select subset of data for the two digits
        mask = np.logical_or(y == digit_a, y == digit_b)
        X_ab = X[mask]
        y_ab = y[mask]
        # optionally subsample to limit dataset size for faster evaluation
        if max_samples is not None and max_samples < len(y_ab):
            # randomly sample max_samples indices
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y_ab), max_samples, replace=False)
            X_ab = X_ab[idx]
            y_ab = y_ab[idx]
        # one‑hot target: [1,0] for digit_a, [0,1] for digit_b
        targets = np.zeros((len(y_ab), 2), dtype=np.float32)
        targets[y_ab == digit_a, 0] = 1.0
        targets[y_ab == digit_b, 1] = 1.0
        # store
        self.X = X_ab.astype(np.float32)
        self.Y = targets

    def loss(self, params: np.ndarray) -> float:
        """Compute mean squared error for the current parameters on this task."""
        total = 0.0
        for x, y in zip(self.X, self.Y):
            # forward pass using all features (no masking)
            out = self.model.forward_given_params(params, x)
            total += np.mean((out - y) ** 2)
        return total / len(self.X)


def build_dense_digit_tasks(n_tasks: int, seed: int = 0) -> Tuple[NNModel, List[BinaryDigitTask]]:
    """Create a sequence of binary digit classification tasks using full features.

    Each task involves two digits.  For example, if n_tasks=5, tasks will be:
    (0 vs 1), (2 vs 3), (4 vs 5), (6 vs 7), (8 vs 9).  If n_tasks < 5, only
    the first n_tasks pairs are used.  If n_tasks > 5, digits will repeat.

    Args:
        n_tasks: Number of tasks to create.
        seed: RNG seed for reproducibility.

    Returns:
        (model, tasks) where model is an NNModel configured for binary
        classification and tasks is a list of BinaryDigitTask objects.
    """
    digits = load_digits()
    X = digits.data / 16.0  # normalize to [0,1]
    y = digits.target
    # create NN model: input_dim=64 (pixels), two hidden layers, output_dim=2 (binary)
    model = NNModel(input_dim=64, hidden_dim1=64, hidden_dim2=32, output_dim=2)
    # assign digit pairs
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks: List[BinaryDigitTask] = []
    for i in range(n_tasks):
        pair = pairs[i % len(pairs)]
        tasks.append(BinaryDigitTask(model, pair[0], pair[1], X, y, max_samples=200))
    return model, tasks


def run_dense_digit_experiment(n_tasks: int = 5, n_evals: int = 300, n_runs: int = 1) -> Tuple[List[float], List[float], List[List[float]]]:
    """Run the dense digit experiment for SGM models.

    Args:
        n_tasks: Number of sequential tasks.
        n_evals: Evaluation budget per task.
        n_runs: Number of independent runs.

    Returns:
        (baseline_ret_means, locking_ret_means, locking_locked_fracs)
        where baseline_ret_means is a list of mean retention ratios on tasks 0..n_tasks-2
        for the baseline, locking_ret_means is the same for the locked variant, and
        locking_locked_fracs is a nested list of locked fraction sequences per run.
    """
    baseline_ret_all = []
    locking_ret_all = []
    locked_fracs_runs: List[List[float]] = []
    for run in range(n_runs):
        np.random.seed(10007 * run)
        model, tasks = build_dense_digit_tasks(n_tasks)
        param_dim = model.total_params
        # baseline
        base_model = SGMBaselineModel(param_dim)
        during_losses = []
        for task in tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            during_losses.append(base_model.best_loss)
        # compute final losses using the best parameters discovered during search.  
        # best_params should always be set after at least one evaluation, but guard
        # against None in extremely small evaluation budgets.  If best_params is
        # None, fallback to the first individual in the population.
        params_to_use = base_model.best_params
        if params_to_use is None:
            # fallback: use the first individual if available
            params_to_use = base_model.pop[0] if base_model.pop is not None else np.zeros(param_dim)
        final_losses = [task.loss(params_to_use) for task in tasks]
        ratios = [final_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        baseline_ret_all.append(ratios)
        # locked
        lock_model = SGMWithLockingModel(param_dim)
        during_losses_l = []
        locked_fracs = []
        for task in tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            during_losses_l.append(lock_model.best_loss)
            locked_fracs.append(float(np.mean(lock_model.lock)))
        # compute final losses for the locked model using best parameters; fallback
        # to a representative individual if best_params is None (unlikely if n_evals > 0)
        params_to_use_l = lock_model.best_params
        if params_to_use_l is None:
            params_to_use_l = lock_model.pop[0] if lock_model.pop is not None else np.zeros(param_dim)
        final_losses_l = [task.loss(params_to_use_l) for task in tasks]
        ratios_l = [final_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        locking_ret_all.append(ratios_l)
        locked_fracs_runs.append(locked_fracs)
    baseline_ret_means = list(np.mean(baseline_ret_all, axis=0))
    locking_ret_means = list(np.mean(locking_ret_all, axis=0))
    return baseline_ret_means, locking_ret_means, locked_fracs_runs


def main():
    n_tasks = 5
    n_evals = 300
    n_runs = 2
    baseline_ret, locking_ret, locked_fracs_runs = run_dense_digit_experiment(
        n_tasks=n_tasks, n_evals=n_evals, n_runs=n_runs
    )
    print("\nDense Digit Tasks (full input) with 64‑64‑32‑2 NN model")
    print(f"Baseline retention ratios per task: {[f'{r:.2f}' for r in baseline_ret]}")
    print(f"Locked retention ratios per task:   {[f'{r:.2f}' for r in locking_ret]}")
    # mean locked fraction across runs
    locked_fracs_mean = list(np.mean(locked_fracs_runs, axis=0))
    for i, frac in enumerate(locked_fracs_mean, start=1):
        print(f"  After task {i}: locked fraction = {frac:.2f}")


if __name__ == '__main__':
    main()