"""
SGM Quantization Tests
======================

This script evaluates the effect of post-training parameter quantization on
the Sparse Gradient Mutation (SGM) coalition-locking primitive. It uses
the feed-forward neural network from ``sgm_model_tests`` and trains both
baseline and locked models on a sequence of synthetic regression tasks.

After training, the best parameters found by each model are quantized to
``n_bits`` bits using uniform quantization across their dynamic range.
Retention ratios (final loss divided by the loss achieved during training)
are compared between the original and quantized versions. Lower ratios
indicate better memory of previous tasks (1.0 means perfect retention).

Usage: run this file directly with ``python3 sgm_quantization_tests.py``.
"""

import numpy as np
from typing import List, Tuple

from sgm_model_tests import (
    NNModel,
    TransformerModel,
    SGMBaselineModel,
    SGMWithLockingModel,
    build_tasks_for_model,
)


def quantize_params(params: np.ndarray, n_bits: int) -> np.ndarray:
    """Uniformly quantize parameters to ``n_bits`` precision.

    Args:
        params: 1D numpy array of floating-point parameters.
        n_bits: Number of bits for quantization (e.g. 8 for 8-bit).

    Returns:
        Quantized and dequantized parameters as a float array.
    """
    if n_bits <= 0 or n_bits > 16:
        raise ValueError("n_bits must be between 1 and 16")
    # compute symmetric range
    max_abs = np.max(np.abs(params))
    if max_abs == 0:
        return params.copy()
    q_levels = 2 ** n_bits - 1
    # map params from [-max_abs, max_abs] to [0, q_levels]
    scaled = (params / (2 * max_abs)) + 0.5  # in [0,1]
    quantized = np.round(scaled * q_levels)  # integer levels
    # dequantize back to float
    dequant = (quantized / q_levels - 0.5) * 2 * max_abs
    return dequant.astype(params.dtype)


def run_quantized_scenario(
    model_type: str = "nn",
    n_tasks: int = 3,
    n_evals: int = 200,
    n_runs: int = 1,
    n_bits: int = 8,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Run baseline and locked SGM with and without quantization.

    Args:
        model_type: "nn" for feed-forward NN or "transformer".
        n_tasks: Number of sequential tasks to learn.
        n_evals: Mutation evaluations per task.
        n_runs: Independent runs for averaging.
        n_bits: Bit-width for uniform quantization.

    Returns:
        A tuple of four lists: (base_ret, base_ret_q, lock_ret, lock_ret_q).
        Each list has length n_runs and contains the average retention ratio
        across tasks (excluding the last task).
    """
    base_ret = []
    base_ret_q = []
    lock_ret = []
    lock_ret_q = []
    for run in range(n_runs):
        rng = np.random.default_rng(run)
        if model_type == "nn":
            model = NNModel(input_dim=64, hidden_dim1=32, hidden_dim2=16, output_dim=10)
        elif model_type == "transformer":
            model = TransformerModel(input_dim=64, hidden_dim=32, output_dim=10)
        else:
            raise ValueError("Unsupported model_type: {}".format(model_type))
        tasks = build_tasks_for_model(model, n_tasks, seed=run)
        param_dim = model.total_params
        # baseline
        base_model = SGMBaselineModel(param_dim)
        during_losses = []
        for task in tasks:
            base_model.reset()
            base_model.step(task, n_evals)
            during_losses.append(base_model.best_loss)
        # compute retention for original params
        final_losses = [task.loss(base_model.best_params) for task in tasks]
        ratios = [final_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        base_ret.append(float(np.mean(ratios)))
        # compute retention for quantized params
        q_params = quantize_params(base_model.best_params, n_bits)
        q_losses = [task.loss(q_params) for task in tasks]
        ratios_q = [q_losses[i] / during_losses[i] if during_losses[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        base_ret_q.append(float(np.mean(ratios_q)))
        # locked
        lock_model = SGMWithLockingModel(param_dim)
        during_losses_l = []
        for task in tasks:
            lock_model.reset()
            lock_model.step(task, n_evals)
            during_losses_l.append(lock_model.best_loss)
        final_losses_l = [task.loss(lock_model.best_params) for task in tasks]
        ratios_l = [final_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        lock_ret.append(float(np.mean(ratios_l)))
        # quantize locked params
        q_params_l = quantize_params(lock_model.best_params, n_bits)
        q_losses_l = [task.loss(q_params_l) for task in tasks]
        ratios_l_q = [q_losses_l[i] / during_losses_l[i] if during_losses_l[i] > 0 else 1.0 for i in range(n_tasks - 1)]
        lock_ret_q.append(float(np.mean(ratios_l_q)))
    return base_ret, base_ret_q, lock_ret, lock_ret_q


def main():
    n_tasks = 3
    n_evals = 150  # limited evaluations for quick tests
    n_runs = 2
    n_bits = 8
    # Test feed-forward NN
    base, base_q, lock, lock_q = run_quantized_scenario(
        model_type="nn", n_tasks=n_tasks, n_evals=n_evals, n_runs=n_runs, n_bits=n_bits
    )
    print("\nFeed-forward NN quantization test ({}-bit)".format(n_bits))
    print("Baseline retention (float): {:.2f} +/- {:.2f}".format(np.mean(base), np.std(base)))
    print("Baseline retention (quantized): {:.2f} +/- {:.2f}".format(np.mean(base_q), np.std(base_q)))
    print("Locked retention (float): {:.2f} +/- {:.2f}".format(np.mean(lock), np.std(lock)))
    print("Locked retention (quantized): {:.2f} +/- {:.2f}".format(np.mean(lock_q), np.std(lock_q)))
    # Optionally test transformer-like model
    base_tr, base_tr_q, lock_tr, lock_tr_q = run_quantized_scenario(
        model_type="transformer", n_tasks=n_tasks, n_evals=n_evals, n_runs=n_runs, n_bits=n_bits
    )
    print("\nTransformer-like quantization test ({}-bit)".format(n_bits))
    print("Baseline retention (float): {:.2f} +/- {:.2f}".format(np.mean(base_tr), np.std(base_tr)))
    print("Baseline retention (quantized): {:.2f} +/- {:.2f}".format(np.mean(base_tr_q), np.std(base_tr_q)))
    print("Locked retention (float): {:.2f} +/- {:.2f}".format(np.mean(lock_tr), np.std(lock_tr)))
    print("Locked retention (quantized): {:.2f} +/- {:.2f}".format(np.mean(lock_tr_q), np.std(lock_tr_q)))


if __name__ == "__main__":
    main()