"""
Model-Based SGM Tests
=====================

This script evaluates the Sparse Gradient Mutation (SGM) primitive with
coalition locking on two more complex architectures: a simple
feed-forward neural network (NNModel) and a simplified transformer-like
model (TransformerModel).  Each architecture has its weights flattened
into a 1D parameter vector; the SGM algorithm performs random
mutations on this parameter vector when learning each task, and
coalition locking freezes parameters deemed causally important.

Test tasks are synthetic regression problems with sparse input masks.
Each task uses a distinct portion of the input dimension and has a
unique target vector.  For a given model and set of tasks the script
runs both a baseline (no locking) and the locked SGM and reports the
average retention ratio across tasks.  A retention ratio near 1.0
indicates perfect memory of earlier tasks; higher values indicate
forgetting.

Usage: `python sgm_model_tests.py`

"""

import numpy as np
from typing import List, Tuple


class NNModel:
    """Simple feed-forward neural network with two hidden layers."""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        # compute parameter sizes
        self.shapes = [
            (input_dim, hidden_dim1),  # W1
            (hidden_dim1,),            # b1
            (hidden_dim1, hidden_dim2),# W2
            (hidden_dim2,),            # b2
            (hidden_dim2, output_dim), # W3
            (output_dim,),             # b3
        ]
        # compute start/end indices for each parameter
        self.starts = []
        self.ends = []
        idx = 0
        for shape in self.shapes:
            n = int(np.prod(shape))
            self.starts.append(idx)
            self.ends.append(idx + n)
            idx += n
        self.total_params = idx

    def forward_given_params(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Forward pass for a batch x given flat parameter vector params."""
        # unpack parameters
        assert params.shape[0] == self.total_params
        W1 = params[self.starts[0]:self.ends[0]].reshape(self.shapes[0])
        b1 = params[self.starts[1]:self.ends[1]].reshape(self.shapes[1])
        W2 = params[self.starts[2]:self.ends[2]].reshape(self.shapes[2])
        b2 = params[self.starts[3]:self.ends[3]].reshape(self.shapes[3])
        W3 = params[self.starts[4]:self.ends[4]].reshape(self.shapes[4])
        b3 = params[self.starts[5]:self.ends[5]].reshape(self.shapes[5])
        # forward
        h1 = np.tanh(x @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out


class TransformerModel:
    """Simplified transformer-like model using elementwise self-attention."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # parameters: W_q, W_k, W_v, W_o, W2 (no biases for simplicity)
        self.shapes = [
            (input_dim, hidden_dim),  # W_q
            (input_dim, hidden_dim),  # W_k
            (input_dim, hidden_dim),  # W_v
            (hidden_dim, hidden_dim), # W_o
            (hidden_dim, output_dim), # W2
        ]
        self.starts = []
        self.ends = []
        idx = 0
        for shape in self.shapes:
            n = int(np.prod(shape))
            self.starts.append(idx)
            self.ends.append(idx + n)
            idx += n
        self.total_params = idx

    def forward_given_params(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Forward pass given flat parameters and input x."""
        assert params.shape[0] == self.total_params
        W_q = params[self.starts[0]:self.ends[0]].reshape(self.shapes[0])
        W_k = params[self.starts[1]:self.ends[1]].reshape(self.shapes[1])
        W_v = params[self.starts[2]:self.ends[2]].reshape(self.shapes[2])
        W_o = params[self.starts[3]:self.ends[3]].reshape(self.shapes[3])
        W2  = params[self.starts[4]:self.ends[4]].reshape(self.shapes[4])
        # compute query, key, value
        q = np.tanh(x @ W_q)
        k = np.tanh(x @ W_k)
        v = np.tanh(x @ W_v)
        # elementwise attention: product of q and k, then elementwise multiply with v
        att = q * k * v  # shape (hidden_dim,)
        h = np.tanh(att @ W_o)         # shape (hidden_dim,)
        out = h @ W2                   # shape (output_dim,)
        return out


class ModelTask:
    """Synthetic regression task for a given model with masked inputs."""

    def __init__(self, model, input_mask: np.ndarray, output_target: np.ndarray, n_samples: int = 1, seed: int = 0) -> None:
        """
        Args:
            model: Model instance (NNModel or TransformerModel).
            input_mask: Boolean array of shape (input_dim,) specifying active input dimensions.
            output_target: Target output vector (shape (output_dim,)).
            n_samples: Number of random input samples to average loss over.
            seed: Seed for generating samples.
        """
        self.model = model
        self.input_mask = input_mask.astype(bool)
        self.output_target = output_target.astype(np.float32)
        # pre-generate input samples
        rng = np.random.default_rng(seed)
        self.inputs = []
        for _ in range(n_samples):
            x = rng.standard_normal(model.input_dim).astype(np.float32)
            # apply mask: only active dims are non-zero
            x_masked = x * self.input_mask.astype(np.float32)
            self.inputs.append(x_masked)

    def loss(self, params: np.ndarray) -> float:
        losses = []
        for x in self.inputs:
            out = self.model.forward_given_params(params, x)
            losses.append(np.mean((out - self.output_target) ** 2))
        return float(np.mean(losses))


class SGMBaselineModel:
    """SGM evolutionary search for a given model, no locking."""

    def __init__(self, param_dim: int) -> None:
        self.param_dim = param_dim
        self.pop = None
        self.best_params = None
        self.best_loss = float('inf')

    def step(self, task: ModelTask, n_evals: int) -> float:
        """Perform search on task for a given number of evaluations."""
        if self.pop is None:
            self.pop = [np.random.randn(self.param_dim) * 0.1 for _ in range(30)]
            self.best_params = self.pop[0].copy()
        fitness = [task.loss(p) for p in self.pop]
        evals = len(self.pop)
        while evals < n_evals:
            elite_indices = np.argsort(fitness)[:6]
            elite = [self.pop[i].copy() for i in elite_indices]
            if fitness[elite_indices[0]] < self.best_loss:
                self.best_loss = fitness[elite_indices[0]]
                self.best_params = self.pop[elite_indices[0]].copy()
            new_pop = list(elite)
            while len(new_pop) < 30 and evals < n_evals:
                parent = elite[np.random.randint(len(elite))]
                child = parent + np.random.randn(self.param_dim) * 0.05
                new_pop.append(child)
                evals += 1
            self.pop = new_pop
            fitness = [task.loss(p) for p in self.pop]
        return self.best_loss

    def reset(self) -> None:
        self.pop = None
        self.best_params = None
        self.best_loss = float('inf')


class SGMWithLockingModel:
    """SGM evolutionary search with coalition locking for model parameters."""

    def __init__(self, param_dim: int) -> None:
        self.param_dim = param_dim
        self.lock = np.zeros(param_dim, dtype=bool)
        self.causal_sum = np.zeros(param_dim, dtype=np.float32)
        self.causal_count = np.ones(param_dim, dtype=np.float32)
        self.group_credits = np.zeros(param_dim, dtype=np.int32)
        self.pop = None
        self.best_params = None
        self.best_loss = float('inf')

    def init_pop(self) -> None:
        self.pop = []
        for _ in range(30):
            x = np.random.randn(self.param_dim) * 0.1
            if self.best_params is not None:
                x[self.lock] = self.best_params[self.lock]
            self.pop.append(x)

    def measure_causality(self, elite: List[np.ndarray], task: ModelTask) -> None:
        x = elite[0]
        base = task.loss(x)
        for d in range(self.param_dim):
            if self.lock[d]:
                continue
            x_test = x.copy()
            x_test[d] = 0.0
            score = task.loss(x_test) - base
            self.causal_sum[d] += score
            self.causal_count[d] += 1
        avg_causal = self.causal_sum / self.causal_count
        candidates = [d for d in range(self.param_dim) if (not self.lock[d]) and (0.0 < avg_causal[d] < 1e-4)]
        if len(candidates) >= 3:
            for _ in range(30):
                k = min(5, len(candidates))
                group = list(np.random.choice(candidates, k, replace=False))
                x_test = x.copy()
                x_test[group] = 0.0
                if task.loss(x_test) - base > 3e-4:
                    for d in group:
                        self.group_credits[d] += 1

    def update_locks(self, elite: List[np.ndarray], task: ModelTask) -> None:
        self.measure_causality(elite, task)
        elite_arr = np.array(elite)
        for d in range(self.param_dim):
            if self.lock[d]:
                continue
            var = np.var(elite_arr[:, d])
            avg_causal = self.causal_sum[d] / self.causal_count[d]
            if var < 0.05:
                if avg_causal > 5e-5 or self.group_credits[d] >= 2:
                    self.lock[d] = True

    def step(self, task: ModelTask, n_evals: int) -> float:
        if self.pop is None:
            self.init_pop()
        fitness = [task.loss(p) for p in self.pop]
        evals = len(self.pop)
        while evals < n_evals:
            elite_indices = np.argsort(fitness)[:6]
            elite = [self.pop[i].copy() for i in elite_indices]
            if fitness[elite_indices[0]] < self.best_loss:
                self.best_loss = fitness[elite_indices[0]]
                self.best_params = self.pop[elite_indices[0]].copy()
            self.update_locks(elite, task)
            mutable = np.where(~self.lock)[0]
            if len(mutable) < 5:
                mutable = np.arange(self.param_dim)
            new_pop = list(elite)
            while len(new_pop) < 30 and evals < n_evals:
                parent = elite[np.random.randint(len(elite))]
                child = parent.copy()
                for d in np.random.choice(mutable, min(5, len(mutable)), replace=False):
                    child[d] += np.random.randn() * 0.05
                new_pop.append(child)
                evals += 1
            self.pop = new_pop
            fitness = [task.loss(p) for p in self.pop]
        return self.best_loss

    def reset(self) -> None:
        self.pop = None
        self.best_params = None
        self.best_loss = float('inf')


def run_model_scenario(model_name: str, model, tasks: List[ModelTask], n_evals: int, n_runs: int) -> Tuple[List[float], List[float]]:
    """Run SGM baseline and locking for a given model and tasks."""
    baseline_rets = []
    locking_rets = []
    param_dim = model.total_params
    for run in range(n_runs):
        np.random.seed(run * 123 + param_dim)
        # baseline
        base_model = SGMBaselineModel(param_dim)
        during_losses = []
        for task in tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            during_losses.append(base_model.best_loss)
        final_losses = [task.loss(base_model.best_params) for task in tasks]
        ratios = []
        for i in range(len(tasks) - 1):
            init = during_losses[i]
            fin = final_losses[i]
            ratios.append(fin / init if init > 0 else 1.0)
        baseline_rets.append(np.mean(ratios))
        # locking
        lock_model = SGMWithLockingModel(param_dim)
        during_losses_l = []
        for task in tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            during_losses_l.append(lock_model.best_loss)
        final_losses_l = [task.loss(lock_model.best_params) for task in tasks]
        ratios_l = []
        for i in range(len(tasks) - 1):
            init = during_losses_l[i]
            fin = final_losses_l[i]
            ratios_l.append(fin / init if init > 0 else 1.0)
        locking_rets.append(np.mean(ratios_l))
    return baseline_rets, locking_rets


def build_tasks_for_model(model, n_tasks: int, seed: int = 0) -> List[ModelTask]:
    """Create a list of ModelTask objects with non-overlapping input masks."""
    input_dim = model.input_dim
    output_dim = model.output_dim
    tasks = []
    rng = np.random.default_rng(seed)
    segment = input_dim // n_tasks
    for i in range(n_tasks):
        mask = np.zeros(input_dim, dtype=bool)
        start = i * segment
        end = (i + 1) * segment if i < n_tasks - 1 else input_dim
        mask[start:end] = True
        target = rng.standard_normal(output_dim).astype(np.float32)
        tasks.append(ModelTask(model, mask, target, n_samples=1, seed=seed + i))
    return tasks


def main():
    n_tasks = 3
    n_runs = 2  # reduce for runtime
    n_evals = 500
    # Test feed-forward NN model
    nn_model = NNModel(input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=10)
    nn_tasks = build_tasks_for_model(nn_model, n_tasks, seed=0)
    nn_baseline, nn_locking = run_model_scenario("nn", nn_model, nn_tasks, n_evals, n_runs)
    print("\nFeed-forward NN Model Results")
    print(f"Baseline retention: {np.mean(nn_baseline):.2f} ± {np.std(nn_baseline):.2f}")
    print(f"Locking retention: {np.mean(nn_locking):.2f} ± {np.std(nn_locking):.2f}")
    print(f"Improvement: {(np.mean(nn_baseline) - np.mean(nn_locking)) / np.mean(nn_baseline) * 100:+.1f}%")
    # Test transformer-like model
    transformer_model = TransformerModel(input_dim=64, hidden_dim=32, output_dim=10)
    transformer_tasks = build_tasks_for_model(transformer_model, n_tasks, seed=42)
    tr_baseline, tr_locking = run_model_scenario("transformer", transformer_model, transformer_tasks, n_evals, n_runs)
    print("\nTransformer-like Model Results")
    print(f"Baseline retention: {np.mean(tr_baseline):.2f} ± {np.std(tr_baseline):.2f}")
    print(f"Locking retention: {np.mean(tr_locking):.2f} ± {np.std(tr_locking):.2f}")
    print(f"Improvement: {(np.mean(tr_baseline) - np.mean(tr_locking)) / np.mean(tr_baseline) * 100:+.1f}%")


if __name__ == "__main__":
    main()