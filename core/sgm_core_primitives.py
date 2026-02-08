"""
SGM Rigorous Scientific Testing
===============================

This script implements a suite of additional tests for the Sparse Gradient
Mutation (SGM) coalition locking primitive.  It is designed to probe
the behaviour of the locking mechanism under more challenging and varied
conditions than the simple non‑overlapping tasks used in the original
demo.  The key scenarios include:

1. **Non‑overlapping tasks**: each task uses a distinct contiguous region
   of the parameter vector.  This is the easy case where tasks do not
   share any dimensions and should show near‑perfect retention when
   locking is enabled.
2. **Partially overlapping tasks**: tasks occupy sliding windows in the
   parameter space so that each task overlaps with the previous one.
   This tests whether the coalition detector can identify and lock the
   shared subspace to preserve earlier tasks.
3. **Random masks**: each task uses a randomly selected set of
   dimensions.  Masks are generated with a fixed active fraction.  The
   randomness introduces unpredictable overlaps between tasks.
4. **Contradictory tasks**: a pair of tasks operate on the exact same
   dimensions but have opposite target outputs.  After training both
   sequentially, retention on the first task is measured.  This is an
   adversarial case designed to elicit catastrophic forgetting if the
   lock fails.

For each scenario we compare two models:

* **Baseline (SGMBaseline)**: an evolutionary search that mutates all
  free parameters for every task.  It uses the best loss achieved
  during training for each task as a reference.
* **Locking (SGMWithLocking)**: identical to the baseline but with
  parameter locking enabled.  Once a dimension is identified as
  causally important (using ablation and coalition tests), it is
  permanently frozen.

Both models are trained for the same number of mutation evaluations per
task.  After completing all tasks in a sequence, the scripts compute
the retention ratio for each task except the final one.  The retention
ratio is defined as the final loss on that task divided by the best
loss achieved during its own training.  A ratio of 1.0 indicates
perfect retention; higher values indicate forgetting.

The script prints results for each scenario, including mean and
standard deviation across multiple runs, and summarises the
improvement obtained by locking relative to the baseline.
"""

import numpy as np
from typing import List, Tuple, Dict


class SparseRegionTask:
    """Task with activity in a contiguous dimension range."""

    def __init__(self, input_dim: int, active_region: Tuple[float, float], seed: int = None) -> None:
        """
        Args:
            input_dim: Number of dimensions of the input vector.
            active_region: Tuple (start_frac, end_frac) indicating the fraction
                of the vector that is active for this task.  start_frac
                inclusive, end_frac exclusive.
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.input_dim = input_dim
        start = int(input_dim * active_region[0])
        end = int(input_dim * active_region[1])
        self.active_indices = np.arange(start, end)
        self.W = np.zeros((input_dim, 32), dtype=np.float32)
        # initialise weights only on active indices
        self.W[start:end] = np.random.randn(end - start, 32).astype(np.float32) * 0.1
        self.target = (np.random.randn(32) * 0.3).astype(np.float32)

    def loss(self, x: np.ndarray) -> float:
        """Mean squared error between network output and target."""
        out = np.tanh(x @ self.W)
        return float(np.mean((out - self.target) ** 2))


class MaskedRegionTask:
    """Task with activity on an arbitrary set of dimensions."""

    def __init__(self, input_dim: int, mask: np.ndarray, seed: int = None) -> None:
        """
        Args:
            input_dim: Number of dimensions of the input vector.
            mask: Boolean array of shape (input_dim,) indicating which
                dimensions are active for this task.
            seed: Optional random seed.
        """
        if seed is not None:
            np.random.seed(seed)
        self.input_dim = input_dim
        self.active_indices = np.where(mask)[0]
        self.W = np.zeros((input_dim, 32), dtype=np.float32)
        # initialize weights on active indices
        self.W[self.active_indices] = np.random.randn(len(self.active_indices), 32).astype(np.float32) * 0.1
        self.target = (np.random.randn(32) * 0.3).astype(np.float32)

    def loss(self, x: np.ndarray) -> float:
        out = np.tanh(x @ self.W)
        return float(np.mean((out - self.target) ** 2))


class SGMBaseline:
    """No locking – pure evolutionary search."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.pop = None
        self.best_x = None
        self.best_loss = float('inf')

    def step(self, task, n_evals: int) -> float:
        """Perform one evolutionary search step for a given task.

        Args:
            task: The task providing the loss function.
            n_evals: Number of mutation evaluations.

        Returns:
            The best loss encountered during this step.
        """
        if self.pop is None:
            # initialise population of candidate solutions
            self.pop = [np.random.randn(self.dim) * 0.3 for _ in range(30)]
            self.best_x = self.pop[0].copy()

        fitness = [task.loss(p) for p in self.pop]
        evals = len(self.pop)

        while evals < n_evals:
            # select elites
            elite_indices = np.argsort(fitness)[:6]
            elite = [self.pop[i].copy() for i in elite_indices]
            # update global best
            if fitness[elite_indices[0]] < self.best_loss:
                self.best_loss = fitness[elite_indices[0]]
                self.best_x = self.pop[elite_indices[0]].copy()
            # generate new population by mutating elites
            new_pop = list(elite)
            while len(new_pop) < 30 and evals < n_evals:
                parent = elite[np.random.randint(len(elite))]
                child = parent + np.random.randn(self.dim) * 0.1
                new_pop.append(child)
                evals += 1
            self.pop = new_pop
            fitness = [task.loss(p) for p in self.pop]
        return self.best_loss

    def reset(self) -> None:
        """Reset the internal state before a new task."""
        self.pop = None
        self.best_loss = float('inf')


class SGMWithLocking:
    """Coalition locking using causal importance and coalition tests."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.lock = np.zeros(dim, dtype=bool)
        self.causal_sum = np.zeros(dim, dtype=np.float32)
        self.causal_count = np.ones(dim, dtype=np.float32)
        self.group_credits = np.zeros(dim, dtype=np.int32)
        self.pop = None
        self.best_x = None
        self.best_loss = float('inf')

    def init_pop(self) -> None:
        """Initialise population for a new task."""
        self.pop = []
        for _ in range(30):
            x = np.random.randn(self.dim) * 0.3
            # copy locked weights from best_x if available
            if self.best_x is not None:
                x[self.lock] = self.best_x[self.lock]
            self.pop.append(x)

    def measure_causality(self, elite: List[np.ndarray], task) -> None:
        """Update causal metrics based on ablation and coalition tests."""
        x = elite[0]
        base = task.loss(x)
        # single-dim ablation
        for d in range(self.dim):
            if self.lock[d]:
                continue
            x_test = x.copy()
            x_test[d] = 0.0
            score = task.loss(x_test) - base
            self.causal_sum[d] += score
            self.causal_count[d] += 1.0
        # compute average causal importance
        avg_causal = self.causal_sum / self.causal_count
        # candidate dims: small positive causal score and not locked
        candidates = [d for d in range(self.dim)
                      if (not self.lock[d]) and (0.0 < avg_causal[d] < 1e-4)]
        # coalition tests: ablate small groups of weak candidates
        if len(candidates) >= 3:
            for _ in range(30):
                k = min(5, len(candidates))
                group = list(np.random.choice(candidates, k, replace=False))
                x_test = x.copy()
                x_test[group] = 0.0
                if task.loss(x_test) - base > 3e-4:
                    for d in group:
                        self.group_credits[d] += 1

    def update_locks(self, elite: List[np.ndarray], task) -> None:
        """Determine which dims to lock after evaluating elites."""
        self.measure_causality(elite, task)
        elite_arr = np.array(elite)
        for d in range(self.dim):
            if self.lock[d]:
                continue
            var = np.var(elite_arr[:, d])
            avg_causal = self.causal_sum[d] / self.causal_count[d]
            if var < 0.15:
                if avg_causal > 5e-5 or self.group_credits[d] >= 2:
                    self.lock[d] = True

    def step(self, task, n_evals: int) -> float:
        """Perform one evolutionary search step with locking enabled."""
        if self.pop is None:
            self.init_pop()
        fitness = [task.loss(p) for p in self.pop]
        evals = len(self.pop)
        while evals < n_evals:
            elite_indices = np.argsort(fitness)[:6]
            elite = [self.pop[i].copy() for i in elite_indices]
            # update best
            if fitness[elite_indices[0]] < self.best_loss:
                self.best_loss = fitness[elite_indices[0]]
                self.best_x = self.pop[elite_indices[0]].copy()
            # update locks based on current elite
            self.update_locks(elite, task)
            # determine mutable dims
            mutable = np.where(~self.lock)[0]
            if len(mutable) < 5:
                mutable = np.arange(self.dim)
            # generate new population by mutating only mutable dims
            new_pop = list(elite)
            while len(new_pop) < 30 and evals < n_evals:
                parent = elite[np.random.randint(len(elite))]
                child = parent.copy()
                for d in np.random.choice(mutable, min(5, len(mutable)), replace=False):
                    child[d] += np.random.randn() * 0.1
                new_pop.append(child)
                evals += 1
            self.pop = new_pop
            fitness = [task.loss(p) for p in self.pop]
        return self.best_loss

    def reset(self) -> None:
        """Reset internal state before a new task."""
        self.pop = None
        self.best_loss = float('inf')


def generate_non_overlap_tasks(dims: int, n_tasks: int, seed: int = 0) -> List[SparseRegionTask]:
    """Generate non‑overlapping tasks splitting the dimensions evenly."""
    tasks = []
    for i in range(n_tasks):
        start = i / n_tasks
        end = (i + 1) / n_tasks
        tasks.append(SparseRegionTask(dims, (start, end), seed=seed + i))
    return tasks


def generate_partial_overlap_tasks(dims: int, n_tasks: int, seed: int = 0) -> List[SparseRegionTask]:
    """Generate tasks with sliding windows overlapping half of each other."""
    tasks = []
    # shift half of window length between tasks
    window = 1.0 / (n_tasks)
    shift = window / 2.0
    for i in range(n_tasks):
        start = i * shift
        end = start + window
        # wrap-around for last tasks if end > 1.0
        if end > 1.0:
            end = 1.0
            start = end - window
        tasks.append(SparseRegionTask(dims, (start, end), seed=seed + i))
    return tasks


def generate_random_mask_tasks(dims: int, n_tasks: int, active_frac: float = 0.2, seed: int = 0) -> List[MaskedRegionTask]:
    """Generate tasks with random masks of active dimensions."""
    tasks = []
    rng = np.random.default_rng(seed)
    for i in range(n_tasks):
        mask = rng.random(dims) < active_frac
        # ensure at least one dimension is active
        if not np.any(mask):
            mask[rng.integers(dims)] = True
        tasks.append(MaskedRegionTask(dims, mask, seed=seed + i))
    return tasks


def generate_contradictory_tasks(dims: int, seed: int = 0) -> List[SparseRegionTask]:
    """Generate two tasks on the same region but with reversed targets."""
    # single region: use half of dims
    start = 0.2
    end = 0.8
    task1 = SparseRegionTask(dims, (start, end), seed=seed)
    task2 = SparseRegionTask(dims, (start, end), seed=seed + 1)
    # flip the target sign for task2 to create contradictory objective
    task2.target = -task2.target
    return [task1, task2]


def run_scenario(tasks: List, dims: int, n_evals: int, n_runs: int) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Run baseline and locking models on a given sequence of tasks.

    Args:
        tasks: List of tasks to train sequentially.
        dims: Dimensionality of the parameter vector.
        n_evals: Number of evaluations per task for the evolutionary search.
        n_runs: Number of independent runs (different seeds).

    Returns:
        Tuple of two dicts mapping model name to list of retention ratios for each run.
    """
    baseline_rets = []
    locking_rets = []
    for run in range(n_runs):
        np.random.seed(run * 100 + dims)
        # baseline
        base_model = SGMBaseline(dims)
        during_losses = []
        for t in tasks:
            base_model.reset()
            base_model.step(t, n_evals)
            during_losses.append(base_model.best_loss)
        final_losses = [t.loss(base_model.best_x) for t in tasks]
        # compute average retention for all but last task
        ratios = []
        for i in range(len(tasks) - 1):
            init = during_losses[i]
            fin = final_losses[i]
            ratios.append(fin / init if init > 0 else 1.0)
        baseline_rets.append(np.mean(ratios))
        # locking
        lock_model = SGMWithLocking(dims)
        during_losses_l = []
        for t in tasks:
            lock_model.reset()
            lock_model.step(t, n_evals)
            during_losses_l.append(lock_model.best_loss)
        final_losses_l = [t.loss(lock_model.best_x) for t in tasks]
        ratios_l = []
        for i in range(len(tasks) - 1):
            init = during_losses_l[i]
            fin = final_losses_l[i]
            ratios_l.append(fin / init if init > 0 else 1.0)
        locking_rets.append(np.mean(ratios_l))
    return {"baseline": baseline_rets}, {"locking": locking_rets}


def main():
    dims_list = [128, 256]
    n_runs = 3
    n_evals = 1000
    scenarios = {
        "non_overlap": generate_non_overlap_tasks,
        "partial_overlap": generate_partial_overlap_tasks,
        "random_mask": generate_random_mask_tasks,
        "contradictory": generate_contradictory_tasks,
    }
    print("\nSGM Rigorous Test Suite Results")
    print("================================")
    for scenario_name, gen_func in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        for dims in dims_list:
            # prepare tasks
            if scenario_name == "contradictory":
                tasks = gen_func(dims, seed=42)
            elif scenario_name == "random_mask":
                tasks = gen_func(dims, n_tasks=4, active_frac=0.2, seed=42)
            else:
                tasks = gen_func(dims, n_tasks=4, seed=42)
            baseline_res, locking_res = run_scenario(tasks, dims, n_evals, n_runs)
            b_mean, b_std = np.mean(baseline_res["baseline"]), np.std(baseline_res["baseline"])
            l_mean, l_std = np.mean(locking_res["locking"]), np.std(locking_res["locking"])
            improvement = (b_mean - l_mean) / b_mean * 100 if b_mean != 0 else 0.0
            print(f"  Dims {dims:4d} | Baseline {b_mean:.2f}±{b_std:.2f} | Locking {l_mean:.2f}±{l_std:.2f} | Improvement {improvement:+6.1f}%")


if __name__ == "__main__":
    main()