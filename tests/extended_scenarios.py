"""
SGM Extended Tests
==================

This script extends the evaluation of the Sparse Gradient Mutation (SGM)
primitive to cover inference-time sparsity, incremental personalization
and parameter scaling on more complex architectures.  The script
leverages models and SGM classes defined in `sgm_model_tests.py` and
adds a hybrid model combining a neural network and a logistic regression
layer.

The tests implemented here include:

1. **Inference-time sparsity measurement**: For a sequence of tasks, we
   record the fraction of dimensions locked after each task.  This
   indicates how many parameters remain mutable and therefore how many
   parameters are actually touched by the optimizer or by inference.

2. **Incremental personalization**: We train a base model on several
   shared tasks, then train user-specific tasks.  We measure retention
   on base tasks and performance on personalization tasks for both
   baseline and locked versions.

3. **Hybrid model test**: We define a model that first processes
   inputs through a neural network and then feeds the hidden
   representation into a logistic regression classifier.  This model
   mixes neural-network and traditional linear ML components.

The results printed by this script illustrate how the coalition
locking primitive behaves under these new scenarios.
"""

import numpy as np
from typing import List, Tuple

from sgm_model_tests import (
    NNModel,
    ModelTask,
    SGMBaselineModel,
    SGMWithLockingModel,
    build_tasks_for_model,
)


class HybridModel:
    """Hybrid model: neural network followed by logistic regression."""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int) -> None:
        # underlying neural network part
        self.nn = NNModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim2)
        # logistic regression weights mapping hidden2 -> output
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        # shapes for logistic layer
        self.lr_shape = (hidden_dim2, output_dim)
        # total params = nn params + logistic params
        self.nn_param_dim = self.nn.total_params
        self.lr_param_dim = int(np.prod(self.lr_shape))
        self.total_params = self.nn_param_dim + self.lr_param_dim
    
    def forward_given_params(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        assert params.shape[0] == self.total_params
        # first part for nn
        nn_params = params[: self.nn_param_dim]
        lr_params = params[self.nn_param_dim :]
        # compute hidden representation
        hidden_rep = self.nn.forward_given_params(nn_params, x)
        # logistic regression output
        W_lr = lr_params.reshape(self.lr_shape)
        out = hidden_rep @ W_lr
        return out


def measure_inference_sparsity(model, tasks: List[ModelTask], n_evals: int, n_runs: int) -> Tuple[List[float], List[float]]:
    """
    Measure the fraction of locked parameters after each task.

    Returns two lists: average locked fraction after each task for the
    baseline (which locks nothing) and for the locked model.
    """
    baseline_locked_frac = []
    locking_locked_frac = []
    param_dim = model.total_params if hasattr(model, 'total_params') else model.nn.total_params + model.lr_param_dim
    for run in range(n_runs):
        np.random.seed(run * 77 + param_dim)
        # baseline: no locks => always 0
        base_model = SGMBaselineModel(param_dim)
        locked = 0
        fracs = []
        for task in tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            fracs.append(0.0)
        baseline_locked_frac.append(fracs)
        # locking
        lock_model = SGMWithLockingModel(param_dim)
        fracs_l = []
        for task in tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            # fraction locked
            frac = float(np.sum(lock_model.lock)) / float(param_dim)
            fracs_l.append(frac)
        locking_locked_frac.append(fracs_l)
    # average across runs
    baseline_avg = np.mean(baseline_locked_frac, axis=0).tolist()
    locking_avg = np.mean(locking_locked_frac, axis=0).tolist()
    return baseline_avg, locking_avg


def run_incremental_personalization(model, base_tasks: List[ModelTask], personal_tasks: List[ModelTask], n_evals: int, n_runs: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Train base tasks, then personalization tasks.  Measure retention on base
    tasks and performance on personal tasks.

    Returns ((base_ret_base, personal_perf_base), (base_ret_lock, personal_perf_lock)).
    """
    param_dim = model.total_params if hasattr(model, 'total_params') else model.nn.total_params + model.lr_param_dim
    base_ret_baseline = []
    personal_perf_baseline = []
    base_ret_lock = []
    personal_perf_lock = []
    for run in range(n_runs):
        np.random.seed(run * 99 + param_dim)
        # baseline
        base_model = SGMBaselineModel(param_dim)
        # train base tasks
        for task in base_tasks:
            base_model.reset()
            base_model.step(task, n_evals)
        # evaluate base tasks after base training
        base_losses_before = [task.loss(base_model.best_params) for task in base_tasks]
        # train personalization tasks sequentially on same model
        for task in personal_tasks:
            base_model.reset()
            base_model.step(task, n_evals)
        # evaluate retention on base tasks
        base_losses_after = [task.loss(base_model.best_params) for task in base_tasks]
        # evaluate performance on personalization tasks
        personal_losses = [task.loss(base_model.best_params) for task in personal_tasks]
        # compute retention ratio and personal performance (mean)
        base_ret_baseline.append(np.mean([after / before if before > 0 else 1.0 for before, after in zip(base_losses_before, base_losses_after)]))
        personal_perf_baseline.append(np.mean(personal_losses))
        # locked model
        lock_model = SGMWithLockingModel(param_dim)
        for task in base_tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
        base_losses_before_l = [task.loss(lock_model.best_params) for task in base_tasks]
        for task in personal_tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
        base_losses_after_l = [task.loss(lock_model.best_params) for task in base_tasks]
        personal_losses_l = [task.loss(lock_model.best_params) for task in personal_tasks]
        base_ret_lock.append(np.mean([after / before if before > 0 else 1.0 for before, after in zip(base_losses_before_l, base_losses_after_l)]))
        personal_perf_lock.append(np.mean(personal_losses_l))
    return ((float(np.mean(base_ret_baseline)), float(np.mean(personal_perf_baseline))), (float(np.mean(base_ret_lock)), float(np.mean(personal_perf_lock))))


def run_parameter_scaling(model_constructor, dims_list: List[int], n_tasks: int, n_evals: int, n_runs: int) -> List[Tuple[int, float, float]]:
    """
    Evaluate retention ratio across increasing parameter dimensions.
    Returns list of tuples (dims, baseline_retention, locked_retention).
    """
    results = []
    for dims in dims_list:
        model = model_constructor(dims)
        tasks = build_tasks_for_model(model, n_tasks, seed=123)
        baseline_rets = []
        locking_rets = []
        param_dim = model.total_params if hasattr(model, 'total_params') else model.nn.total_params + model.lr_param_dim
        for run in range(n_runs):
            np.random.seed(run * 50 + dims)
            # baseline
            base_model = SGMBaselineModel(param_dim)
            during_losses = []
            for t in tasks:
                base_model.reset()
                base_model.step(t, n_evals)
                during_losses.append(base_model.best_loss)
            final_losses = [t.loss(base_model.best_params) for t in tasks]
            ratios = []
            for i in range(len(tasks) - 1):
                init = during_losses[i]
                fin = final_losses[i]
                ratios.append(fin / init if init > 0 else 1.0)
            baseline_rets.append(np.mean(ratios))
            # locking
            lock_model = SGMWithLockingModel(param_dim)
            during_losses_l = []
            for t in tasks:
                lock_model.reset()
                lock_model.step(t, n_evals)
                during_losses_l.append(lock_model.best_loss)
            final_losses_l = [t.loss(lock_model.best_params) for t in tasks]
            ratios_l = []
            for i in range(len(tasks) - 1):
                init = during_losses_l[i]
                fin = final_losses_l[i]
                ratios_l.append(fin / init if init > 0 else 1.0)
            locking_rets.append(np.mean(ratios_l))
        results.append((dims, float(np.mean(baseline_rets)), float(np.mean(locking_rets))))
    return results


def main():
    # Inference sparsity test on NN model
    nn_model = NNModel(input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=8)
    nn_tasks = build_tasks_for_model(nn_model, n_tasks=4, seed=2024)
    base_frac, lock_frac = measure_inference_sparsity(nn_model, nn_tasks, n_evals=300, n_runs=2)
    print("\nInference-time sparsity (NN model)")
    for i, (bf, lf) in enumerate(zip(base_frac, lock_frac)):
        print(f"  After task {i+1}: baseline locked {bf:.2f}, locking locked {lf:.2f}")
    # Incremental personalization on NN model
    base_tasks = build_tasks_for_model(nn_model, n_tasks=2, seed=42)
    personal_tasks = build_tasks_for_model(nn_model, n_tasks=2, seed=84)
    (base_base_ret, base_personal_perf), (lock_base_ret, lock_personal_perf) = run_incremental_personalization(
        nn_model, base_tasks, personal_tasks, n_evals=300, n_runs=2
    )
    print("\nIncremental Personalization (NN model)")
    print(f"  Baseline retention on base tasks: {base_base_ret:.2f}")
    print(f"  Baseline personal task loss: {base_personal_perf:.2f}")
    print(f"  Locked retention on base tasks:   {lock_base_ret:.2f}")
    print(f"  Locked personal task loss:   {lock_personal_perf:.2f}")
    # Hybrid model tests
    hybrid_model = HybridModel(input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=8)
    # tasks for hybrid model
    base_tasks_h = []
    segment = hybrid_model.input_dim // 2
    for i in range(2):
        mask = np.zeros(hybrid_model.input_dim, dtype=bool)
        start = i * segment
        end = (i + 1) * segment
        mask[start:end] = True
        target = np.random.standard_normal(hybrid_model.output_dim).astype(np.float32)
        base_tasks_h.append(ModelTask(hybrid_model, mask, target, n_samples=1, seed=100 + i))
    personal_tasks_h = []
    for i in range(2):
        mask = np.zeros(hybrid_model.input_dim, dtype=bool)
        start = (i * segment) // 2
        end = start + segment
        mask[start:end] = True
        target = np.random.standard_normal(hybrid_model.output_dim).astype(np.float32)
        personal_tasks_h.append(ModelTask(hybrid_model, mask, target, n_samples=1, seed=200 + i))
    (base_base_ret_h, base_personal_perf_h), (lock_base_ret_h, lock_personal_perf_h) = run_incremental_personalization(
        hybrid_model, base_tasks_h, personal_tasks_h, n_evals=300, n_runs=2
    )
    print("\nIncremental Personalization (Hybrid model)")
    print(f"  Baseline retention on base tasks: {base_base_ret_h:.2f}")
    print(f"  Baseline personal task loss: {base_personal_perf_h:.2f}")
    print(f"  Locked retention on base tasks:   {lock_base_ret_h:.2f}")
    print(f"  Locked personal task loss:   {lock_personal_perf_h:.2f}")
    # Parameter scaling on NN model
    print("\nParameter scaling (NN model)")
    def make_nn(dim):
        return NNModel(input_dim=dim, hidden_dim1=dim//2, hidden_dim2=dim//4, output_dim=8)
    dims_list = [32, 64, 128]
    scaling_results = run_parameter_scaling(make_nn, dims_list, n_tasks=3, n_evals=300, n_runs=1)
    for dims, b_ret, l_ret in scaling_results:
        print(f"  {dims} dims | baseline ret {b_ret:.2f} | locked ret {l_ret:.2f}")


if __name__ == "__main__":
    main()