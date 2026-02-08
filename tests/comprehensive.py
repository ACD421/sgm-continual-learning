"""
SGM Comprehensive Demo — All tests active
=========================================
This script assumes that sgm_rigorous_tests.py, fractal_cosetting_experiment.py,
and related modules from the sgm_repo are on your Python path.  It runs the
following tests in sequence and prints summary metrics:

1. Parameter scaling across multiple dimensions.
2. Extreme task count retention.
3. Saturation under minimal free parameters.
4. Contradictory/adversarial tasks.
5. Partial and random overlap tasks.
6. Synthetic hierarchical (CIFAR‑like) tasks.
7. Synthetic bag‑of‑words (natural‑language‑like) tasks.
8. Fractal cosetting retention.
9. Noise perturbation robustness.

Save this script in the same directory as the extracted SGM modules (or adjust
your PYTHONPATH accordingly) and run it with:

    python3 sgm_comprehensive_real.py
"""

import numpy as np

# Import SGM classes from your extracted repository
from sgm_rigorous_tests import SparseRegionTask, SGMBaseline, SGMWithLocking
from fractal_cosetting_experiment import SGMFractalLocking

def parameter_scaling_test():
    print("=== Parameter Scaling Test ===")
    dims_list = [128, 256, 512]
    n_tasks = 4
    n_evals = 300
    for dims in dims_list:
        tasks = [
            SparseRegionTask(dims, (i/n_tasks, (i+1)/n_tasks), seed=i)
            for i in range(n_tasks)
        ]
        baseline = SGMBaseline(dims)
        locking = SGMWithLocking(dims)
        b_ratios = []
        l_ratios = []
        for task in tasks:
            baseline.reset()
            locking.reset()
            baseline.step(task, n_evals)
            locking.step(task, n_evals)
            b_ratios.append(task.loss(baseline.best_x) / baseline.best_loss)
            l_ratios.append(task.loss(locking.best_x) / locking.best_loss)
        print(f"{dims}-dim: baseline {np.mean(b_ratios):.2f}, locking {np.mean(l_ratios):.2f}")

def extreme_task_count_test():
    print("\n=== Extreme Task Count Test ===")
    dims = 128
    n_tasks = 20  # adjust higher if you have more compute
    n_evals = 200
    tasks = [
        SparseRegionTask(dims, (i/n_tasks, (i+1)/n_tasks), seed=i)
        for i in range(n_tasks)
    ]
    baseline = SGMBaseline(dims)
    locking = SGMWithLocking(dims)
    b_ret = []
    l_ret = []
    base_ref = lock_ref = None
    for i, task in enumerate(tasks):
        baseline.reset()
        locking.reset()
        baseline.step(task, n_evals)
        locking.step(task, n_evals)
        if i == 0:
            base_ref = baseline.best_loss
            lock_ref = locking.best_loss
        else:
            b_ret.append(task.loss(baseline.best_x) / base_ref)
            l_ret.append(task.loss(locking.best_x) / lock_ref)
    print(f"Baseline retention after {n_tasks} tasks: {np.mean(b_ret):.2f}")
    print(f"Locking retention after {n_tasks} tasks: {np.mean(l_ret):.2f}")

def saturation_test():
    print("\n=== Saturation Test ===")
    dims = 128
    n_tasks = 10
    n_evals = 200
    tasks = [
        SparseRegionTask(dims, (i * 0.1, (i + 1) * 0.1), seed=i)
        for i in range(n_tasks)
    ]
    locking = SGMWithLocking(dims)
    ref_loss = None
    ratios = []
    for i, task in enumerate(tasks):
        locking.reset()
        locking.step(task, n_evals)
        if i == 0:
            ref_loss = locking.best_loss
        else:
            ratios.append(task.loss(locking.best_x) / ref_loss)
    print(f"Retention across {n_tasks} tasks: {np.mean(ratios):.2f}")
    print(f"Final locked fraction: {np.mean(locking.lock > 0.5):.2f}")

def contradictory_test():
    print("\n=== Contradictory Task Test ===")
    dims = 128
    tasks = [
        SparseRegionTask(dims, (0.25, 0.75), seed=1),
        SparseRegionTask(dims, (0.25, 0.75), seed=2)
    ]
    baseline = SGMBaseline(dims)
    locking = SGMWithLocking(dims)
    b_losses = []
    l_losses = []
    for t in tasks:
        baseline.reset()
        locking.reset()
        baseline.step(t, 300)
        locking.step(t, 300)
        b_losses.append(t.loss(baseline.best_x))
        l_losses.append(t.loss(locking.best_x))
    print("Baseline final losses:", [round(x, 4) for x in b_losses])
    print("Locked final losses:", [round(x, 4) for x in l_losses])

def overlap_tests():
    print("\n=== Overlap Tests (Partial & Random) ===")
    dims = 128
    partial_tasks = [
        SparseRegionTask(dims, (0.00, 0.50), seed=1),
        SparseRegionTask(dims, (0.25, 0.75), seed=2),
        SparseRegionTask(dims, (0.50, 1.00), seed=3)
    ]
    random_tasks = [
        SparseRegionTask(
            dims,
            (np.random.rand() * 0.7, np.random.rand() * 0.7 + 0.3),
            seed=i
        )
        for i in range(3)
    ]
    for name, tasks in [("Partial Overlap", partial_tasks),
                        ("Random Masks", random_tasks)]:
        baseline = SGMBaseline(dims)
        locking = SGMWithLocking(dims)
        base_ref = lock_ref = None
        b_ratios = []
        l_ratios = []
        for i, t in enumerate(tasks):
            baseline.reset()
            locking.reset()
            baseline.step(t, 200)
            locking.step(t, 200)
            if i == 0:
                base_ref = baseline.best_loss
                lock_ref = locking.best_loss
            else:
                b_ratios.append(t.loss(baseline.best_x) / base_ref)
                l_ratios.append(t.loss(locking.best_x) / lock_ref)
        print(f"{name} – baseline {np.mean(b_ratios):.2f}, locked {np.mean(l_ratios):.2f}")

def hierarchical_natty_tests():
    print("\n=== Hierarchical (CIFAR‑like) and Bag‑of‑Words (Natty) Tests ===")
    dims = 150
    # Hierarchical: 3 tasks with overlapping 30-dim windows in a 150-dim vector
    hier_tasks = [
        SparseRegionTask(dims, (i*0.2, i*0.2 + 0.4), seed=i)
        for i in range(3)
    ]
    # Bag-of-Words: random 30-dim masks across 200-dim vocabulary
    vocab = 200
    natty_tasks = []
    for i in range(3):
        start = np.random.randint(0, vocab - 50)
        end = start + np.random.randint(30, 50)
        natty_tasks.append(
            SparseRegionTask(vocab, (start/vocab, end/vocab), seed=100+i)
        )
    for name, tasks in [("Hierarchical", hier_tasks), ("Bag‑of‑Words", natty_tasks)]:
        baseline = SGMBaseline(tasks[0].input_dim)
        locking = SGMWithLocking(tasks[0].input_dim)
        base_ref = lock_ref = None
        b_ratios = []
        l_ratios = []
        for i, t in enumerate(tasks):
            baseline.reset()
            locking.reset()
            baseline.step(t, 200)
            locking.step(t, 200)
            if i == 0:
                base_ref = baseline.best_loss
                lock_ref = locking.best_loss
            else:
                b_ratios.append(t.loss(baseline.best_x) / base_ref)
                l_ratios.append(t.loss(locking.best_x) / lock_ref)
        print(f"{name} – baseline {np.mean(b_ratios):.2f}, locked {np.mean(l_ratios):.2f}")

def fractal_test():
    print("\n=== Fractal Cosetting Test ===")
    dims = 256
    group_size = 16
    tasks = [
        SparseRegionTask(dims, (i/4, (i+1)/4), seed=i)
        for i in range(4)
    ]
    baseline = SGMBaseline(dims)
    fractal = SGMFractalLocking(dims, group_size=group_size)
    base_ref = frac_ref = None
    b_ratios = []
    f_ratios = []
    for i, t in enumerate(tasks):
        baseline.reset()
        fractal.reset()
        baseline.step(t, 300)
        fractal.step(t, 300)
        if i == 0:
            base_ref = baseline.best_loss
            frac_ref = fractal.best_loss
        else:
            b_ratios.append(t.loss(baseline.best_x) / base_ref)
            f_ratios.append(t.loss(fractal.best_x) / frac_ref)
    print(f"Baseline retention: {np.mean(b_ratios):.2f}")
    print(f"Fractal retention: {np.mean(f_ratios):.2f}")

def noise_perturbation_test():
    print("\n=== Noise Perturbation Test ===")
    dims = 128
    task = SparseRegionTask(dims, (0.2, 0.8), seed=42)
    model = SGMWithLocking(dims)
    model.step(task, 300)
    locked_best = model.best_x.copy()
    base_loss = task.loss(locked_best)
    for sigma in [0.1, 0.5, 1.0]:
        noise = np.random.randn(dims) * sigma
        perturbed = locked_best + noise
        new_loss = task.loss(perturbed)
        print(f"σ={sigma:.1f}: loss multiplier {new_loss/base_loss:.2f}×")

if __name__ == "__main__":
    parameter_scaling_test()
    extreme_task_count_test()
    saturation_test()
    contradictory_test()
    overlap_tests()
    hierarchical_natty_tests()
    fractal_test()
    noise_perturbation_test()
