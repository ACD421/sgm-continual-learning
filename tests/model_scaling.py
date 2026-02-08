"""
Extended model experiments for Sparse Gradient Mutation (SGM)
================================================================

This script builds upon the earlier ``sgm_model_tests.py`` and ``sgm_extended_tests.py``
to evaluate the SGM coalition locking primitive in several new scenarios:

1. **Inference‑time sparsity** for neural network (NN) and transformer‑like models.
   For each model, we measure how the fraction of locked dimensions grows as new tasks
   are learned.  This illustrates how much of the parameter space remains active
   during inference for the locked variant compared to a baseline that never locks.

2. **Incremental personalization** for NN and hybrid models.  Two ``base`` tasks
   are learned first, followed by two ``personal`` tasks.  We measure how well
   the base tasks are retained and how well the personal tasks are learned under
   both baseline and locking regimes.

3. **Parameter scaling** for NN and transformer models.  We vary the input/hidden
   dimensionality and observe how the retention ratio changes with model size for
   baseline and locked systems.  The goal is to show that locked models maintain
   near‑constant retention while baselines degrade rapidly.

4. **Hybrid NN+ML model** experiments.  We combine a neural network with a
   logistic regression head to form a hybrid model with both non‑linear and
   linear components.  We then evaluate personalization performance on this
   hybrid architecture.

To run the experiments, execute this file directly:

```
python3 sgm_model_extended_experiments.py
```

Note: these experiments are designed to run quickly (small evaluation budgets
and few runs) to fit within time constraints.  Feel free to increase
``n_evals`` or ``n_runs`` for more robust statistics.
"""

import numpy as np
from typing import List, Tuple

from sgm_model_tests import (
    NNModel,
    TransformerModel,
    SGMBaselineModel,
    SGMWithLockingModel,
    ModelTask,
)

def build_tasks_for_model(model, n_tasks: int, seed: int = 0) -> List[ModelTask]:
    """Create a list of ModelTask objects with non‑overlapping input masks.

    Each task activates a contiguous segment of the input dimensions.  A new
    random target vector is generated for each task.

    Args:
        model: Model instance (NNModel or TransformerModel).
        n_tasks: Number of tasks to create.
        seed: RNG seed for reproducibility.

    Returns:
        List of ModelTask objects.
    """
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


def measure_inference_sparsity_model(model, n_tasks: int = 4, n_evals: int = 300, n_runs: int = 1) -> Tuple[List[float], List[float]]:
    """Measure inference‑time sparsity for a given model.

    Runs ``n_tasks`` sequential synthetic tasks on both a baseline (no locking)
    and the locked SGM model.  After each task, records the fraction of locked
    dimensions (i.e., parameters frozen by coalition locking).  For the baseline
    this fraction is always zero.

    Args:
        model: Instance of NNModel or TransformerModel.
        n_tasks: Number of sequential tasks to train.
        n_evals: Number of evaluations (mutations) per task.
        n_runs: Number of independent repetitions.

    Returns:
        A tuple ``(baseline_locked_fracs, locking_locked_fracs)`` where each
        element is a list of average locked fractions after each task across runs.
    """
    param_dim = model.total_params
    baseline_fracs = np.zeros((n_runs, n_tasks))
    locking_fracs = np.zeros((n_runs, n_tasks))
    for run in range(n_runs):
        np.random.seed(run * 101)
        tasks = build_tasks_for_model(model, n_tasks, seed=run * 17)
        # Baseline: no locking
        base_model = SGMBaselineModel(param_dim)
        for i, task in enumerate(tasks):
            base_model.reset()
            base_model.step(task, n_evals)
            # baseline never locks any parameters
            baseline_fracs[run, i] = 0.0
        # Locked variant
        lock_model = SGMWithLockingModel(param_dim)
        for i, task in enumerate(tasks):
            lock_model.reset()
            lock_model.step(task, n_evals)
            # fraction of locked parameters after this task
            locking_fracs[run, i] = np.mean(lock_model.lock.astype(np.float32))
    return baseline_fracs.mean(axis=0).tolist(), locking_fracs.mean(axis=0).tolist()


def run_incremental_personalization_model(model, n_evals: int = 300, n_runs: int = 1) -> Tuple[float, float, float, float]:
    """Run an incremental personalization scenario for a given model.

    Two ``base`` tasks are trained sequentially, followed by two ``personal``
    tasks.  Retention of base tasks and performance on personal tasks are
    measured for baseline and locked models.

    Args:
        model: Instance of NNModel or TransformerModel (or compatible with ModelTask).
        n_evals: Number of evaluations per task.
        n_runs: Number of independent repetitions.

    Returns:
        Tuple of four floats:
            (baseline_ret, baseline_personal_loss, locked_ret, locked_personal_loss)
    where ``baseline_ret`` is the average retention ratio on the two base tasks
    for the baseline, ``baseline_personal_loss`` is the final loss on the last
    personal task for the baseline, and likewise for the locked model.
    """
    param_dim = model.total_params
    base_rets = []
    base_personal_losses = []
    lock_rets = []
    lock_personal_losses = []
    for run in range(n_runs):
        np.random.seed(run * 257)
        # build four tasks: first two are base tasks, next two are personalized tasks
        tasks = build_tasks_for_model(model, 4, seed=run * 31)
        base_tasks = tasks[:2]
        personal_tasks = tasks[2:]
        # Baseline
        base_model = SGMBaselineModel(param_dim)
        during_losses = []
        # train base tasks sequentially
        for task in base_tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            during_losses.append(base_model.best_loss)
        # evaluate retention on base tasks after training personalized tasks
        # train personalized tasks sequentially (update same baseline model)
        for task in personal_tasks:
            base_model.reset()
            base_model.step(task, n_evals)
        final_losses = [task.loss(base_model.best_params) for task in base_tasks]
        ratios = [final_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(len(base_tasks))]
        base_rets.append(float(np.mean(ratios)))
        base_personal_losses.append(personal_tasks[-1].loss(base_model.best_params))
        # Locked
        lock_model = SGMWithLockingModel(param_dim)
        during_losses_l = []
        for task in base_tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            during_losses_l.append(lock_model.best_loss)
        for task in personal_tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
        final_losses_l = [task.loss(lock_model.best_params) for task in base_tasks]
        ratios_l = [final_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(len(base_tasks))]
        lock_rets.append(float(np.mean(ratios_l)))
        lock_personal_losses.append(personal_tasks[-1].loss(lock_model.best_params))
    return (
        float(np.mean(base_rets)),
        float(np.mean(base_personal_losses)),
        float(np.mean(lock_rets)),
        float(np.mean(lock_personal_losses)),
    )


def run_parameter_scaling_model(model_class, dims_list: List[int], model_kwargs: dict, n_tasks: int = 4, n_evals: int = 300, n_runs: int = 1) -> Tuple[List[float], List[float]]:
    """Run parameter scaling experiments for a given model class.

    Creates models of varying input dimensionality (specified in ``dims_list``) and
    evaluates retention ratios across tasks for baseline and locked systems.

    Args:
        model_class: Class of the model to instantiate (e.g., NNModel or TransformerModel).
        dims_list: List of input dimensions to test.
        model_kwargs: Additional kwargs to pass when constructing the model (besides input_dim).
        n_tasks: Number of tasks for each model.
        n_evals: Evaluation budget per task.
        n_runs: Number of runs per model size.

    Returns:
        Tuple of two lists: baseline retention means and locked retention means for each dimension.
    """
    baseline_means = []
    locking_means = []
    for dim in dims_list:
        # update model_kwargs with new input dimension
        kwargs = dict(model_kwargs)
        kwargs['input_dim'] = dim
        # adjust hidden dimensions if provided as relative values (e.g., half of input dim)
        if 'hidden_dim1' in kwargs and kwargs['hidden_dim1'] is None:
            kwargs['hidden_dim1'] = max(4, dim // 2)
        if 'hidden_dim2' in kwargs and kwargs['hidden_dim2'] is None:
            kwargs['hidden_dim2'] = max(4, dim // 4)
        if 'hidden_dim' in kwargs and kwargs['hidden_dim'] is None:
            kwargs['hidden_dim'] = max(4, dim // 2)
        # instantiate model
        model = model_class(**kwargs)
        # build tasks
        tasks = build_tasks_for_model(model, n_tasks, seed=42 + dim)
        # run baseline and locked scenario
        dim_baseline = []
        dim_locking = []
        for run in range(n_runs):
            np.random.seed(run * 19 + dim)
            # baseline
            base_model = SGMBaselineModel(model.total_params)
            during_losses = []
            for task in tasks:
                base_model.reset()
                base_model.step(task, n_evals)
                during_losses.append(base_model.best_loss)
            final_losses = [task.loss(base_model.best_params) for task in tasks]
            ratios = [final_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(n_tasks - 1)]
            dim_baseline.append(float(np.mean(ratios)))
            # locked
            lock_model = SGMWithLockingModel(model.total_params)
            during_losses_l = []
            for task in tasks:
                lock_model.reset()
                lock_model.step(task, n_evals)
                during_losses_l.append(lock_model.best_loss)
            final_losses_l = [task.loss(lock_model.best_params) for task in tasks]
            ratios_l = [final_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(n_tasks - 1)]
            dim_locking.append(float(np.mean(ratios_l)))
        baseline_means.append(float(np.mean(dim_baseline)))
        locking_means.append(float(np.mean(dim_locking)))
    return baseline_means, locking_means


class HybridModel:
    """Hybrid neural network + linear model for testing SGM.

    This model composes a small NNModel with a logistic regression (linear) head.
    The parameter vector comprises the NN parameters followed by the linear head
    weights.  The forward pass applies the NN to the input and then performs a
    linear combination on the NN output.  This model is used to test whether
    coalition locking works for architectures that mix nonlinear and linear parts.
    """
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, nn_out_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.nn_out_dim = nn_out_dim
        # underlying NN
        self.nn = NNModel(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=nn_out_dim)
        # linear head dimension (one output)
        self.output_dim = 1
        # total parameters = nn parameters + nn_out_dim weights + bias
        self.nn_param_dim = self.nn.total_params
        self.lin_param_dim = nn_out_dim + 1  # weights and bias
        self.total_params = self.nn_param_dim + self.lin_param_dim

    def forward_given_params(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        assert params.shape[0] == self.total_params
        nn_params = params[: self.nn_param_dim]
        lin_params = params[self.nn_param_dim :]
        nn_out = self.nn.forward_given_params(nn_params, x)
        # linear head: weights are first part, bias is last
        w_lin = lin_params[: self.nn_out_dim]
        b_lin = lin_params[self.nn_out_dim]
        out = np.dot(nn_out, w_lin) + b_lin
        return np.array([out])  # shape (1,)


def run_hybrid_personalization(input_dim: int = 64, hidden_dim1: int = 32, hidden_dim2: int = 16, nn_out_dim: int = 8, n_evals: int = 300, n_runs: int = 1) -> Tuple[float, float, float, float]:
    """Run incremental personalization on the hybrid NN+ML model.

    Two base tasks and two personal tasks are learned sequentially.  Retention
    ratios on the base tasks and loss on the last personal task are returned for
    baseline and locked variants.

    Args:
        input_dim: Dimensionality of input vectors.
        hidden_dim1: First hidden layer size for NN.
        hidden_dim2: Second hidden layer size for NN.
        nn_out_dim: Output dimension of the NN portion.
        n_evals: Number of evaluations per task.
        n_runs: Number of independent runs.

    Returns:
        (baseline_ret, baseline_personal_loss, locked_ret, locked_personal_loss)
    """
    model = HybridModel(input_dim, hidden_dim1, hidden_dim2, nn_out_dim)
    param_dim = model.total_params
    base_rets = []
    base_personal_losses = []
    lock_rets = []
    lock_personal_losses = []
    for run in range(n_runs):
        np.random.seed(run * 601)
        tasks = build_tasks_for_model(model.nn, 4, seed=run * 43)
        # adjust tasks to produce ModelTask compatible with hybrid model
        # tasks built from model.nn have input masks and targets; we need to wrap them
        hybrid_tasks = []
        for i, t in enumerate(tasks):
            # create new ModelTask where model is hybrid and input_mask/target re-used
            hybrid_tasks.append(ModelTask(model, t.input_mask, np.array([t.output_target.mean()]), n_samples=1, seed=run * 43 + i))
        base_tasks = hybrid_tasks[:2]
        personal_tasks = hybrid_tasks[2:]
        # baseline
        base_model = SGMBaselineModel(param_dim)
        during_losses = []
        for task in base_tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            during_losses.append(base_model.best_loss)
        for task in personal_tasks:
            base_model.reset()
            base_model.step(task, n_evals)
        final_losses = [task.loss(base_model.best_params) for task in base_tasks]
        ratios = [final_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(len(base_tasks))]
        base_rets.append(float(np.mean(ratios)))
        base_personal_losses.append(personal_tasks[-1].loss(base_model.best_params))
        # locked
        lock_model = SGMWithLockingModel(param_dim)
        during_losses_l = []
        for task in base_tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            during_losses_l.append(lock_model.best_loss)
        for task in personal_tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
        final_losses_l = [task.loss(lock_model.best_params) for task in base_tasks]
        ratios_l = [final_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(len(base_tasks))]
        lock_rets.append(float(np.mean(ratios_l)))
        lock_personal_losses.append(personal_tasks[-1].loss(lock_model.best_params))
    return (
        float(np.mean(base_rets)),
        float(np.mean(base_personal_losses)),
        float(np.mean(lock_rets)),
        float(np.mean(lock_personal_losses)),
    )


def main():
    # Experiment 1: Inference‑time sparsity for NN and transformer
    nn_model = NNModel(input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=8)
    tr_model = TransformerModel(input_dim=64, hidden_dim=32, output_dim=8)
    print("\nInference‑time sparsity (NN model)")
    nn_b_fracs, nn_l_fracs = measure_inference_sparsity_model(nn_model, n_tasks=4, n_evals=300, n_runs=2)
    for i, (b, l) in enumerate(zip(nn_b_fracs, nn_l_fracs), start=1):
        print(f"  After task {i}: baseline locked {b:.2f}, locking locked {l:.2f}")
    print("\nInference‑time sparsity (Transformer model)")
    tr_b_fracs, tr_l_fracs = measure_inference_sparsity_model(tr_model, n_tasks=4, n_evals=300, n_runs=2)
    for i, (b, l) in enumerate(zip(tr_b_fracs, tr_l_fracs), start=1):
        print(f"  After task {i}: baseline locked {b:.2f}, locking locked {l:.2f}")
    # Experiment 2: Incremental personalization for NN and transformer
    print("\nIncremental personalization (NN model)")
    nn_ret, nn_p_loss, nn_lock_ret, nn_lock_p_loss = run_incremental_personalization_model(nn_model, n_evals=300, n_runs=2)
    print(f"  Baseline retention on base tasks: {nn_ret:.2f}\n  Baseline personal task loss: {nn_p_loss:.2f}")
    print(f"  Locked retention on base tasks:   {nn_lock_ret:.2f}\n  Locked personal task loss:   {nn_lock_p_loss:.2f}")
    print("\nIncremental personalization (Transformer model)")
    tr_ret, tr_p_loss, tr_lock_ret, tr_lock_p_loss = run_incremental_personalization_model(tr_model, n_evals=300, n_runs=2)
    print(f"  Baseline retention on base tasks: {tr_ret:.2f}\n  Baseline personal task loss: {tr_p_loss:.2f}")
    print(f"  Locked retention on base tasks:   {tr_lock_ret:.2f}\n  Locked personal task loss:   {tr_lock_p_loss:.2f}")
    # Experiment 3: Parameter scaling for NN and transformer
    dims_list = [16, 32, 64]
    print("\nParameter scaling (NN model)")
    nn_scale_base, nn_scale_lock = run_parameter_scaling_model(NNModel, dims_list, {'hidden_dim1': None, 'hidden_dim2': None, 'output_dim': 4}, n_tasks=3, n_evals=300, n_runs=1)
    for dim, b, l in zip(dims_list, nn_scale_base, nn_scale_lock):
        print(f"  {dim} dims | baseline ret {b:.2f} | locked ret {l:.2f}")
    print("\nParameter scaling (Transformer model)")
    tr_scale_base, tr_scale_lock = run_parameter_scaling_model(TransformerModel, dims_list, {'hidden_dim': None, 'output_dim': 4}, n_tasks=3, n_evals=300, n_runs=1)
    for dim, b, l in zip(dims_list, tr_scale_base, tr_scale_lock):
        print(f"  {dim} dims | baseline ret {b:.2f} | locked ret {l:.2f}")
    # Experiment 4: Hybrid NN+ML model personalization
    print("\nIncremental personalization (Hybrid model)")
    hy_ret, hy_p_loss, hy_lock_ret, hy_lock_p_loss = run_hybrid_personalization(
        input_dim=64, hidden_dim1=32, hidden_dim2=16, nn_out_dim=8, n_evals=300, n_runs=2
    )
    print(f"  Baseline retention on base tasks: {hy_ret:.2f}\n  Baseline personal task loss: {hy_p_loss:.2f}")
    print(f"  Locked retention on base tasks:   {hy_lock_ret:.2f}\n  Locked personal task loss:   {hy_lock_p_loss:.2f}")


if __name__ == "__main__":
    main()