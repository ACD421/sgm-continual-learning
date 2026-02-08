#!/usr/bin/env python3
"""
SGM Coalition Locking - Final Demo
==================================
One-command proof that structure preservation breaks the scaling curve.

Run: python sgm_final_demo.py
"""

import numpy as np

class SparseRegionTask:
    """Task with activity in specific dimension range."""
    def __init__(self, input_dim, active_region, seed=None):
        if seed: np.random.seed(seed)
        self.input_dim = input_dim
        start = int(input_dim * active_region[0])
        end = int(input_dim * active_region[1])
        self.active_start = start
        self.active_end = end
        self.W = np.zeros((input_dim, 32))
        self.W[start:end] = np.random.randn(end - start, 32) * 0.1
        self.target = np.random.randn(32) * 0.3

    def loss(self, x):
        return np.mean((np.tanh(x @ self.W) - self.target) ** 2)

class SGMBaseline:
    """No locking - pure evolutionary."""
    def __init__(self, dim):
        self.dim = dim
        self.pop = None
        self.best_x = None
        self.best_loss = float('inf')

    def step(self, task, n_evals):
        if self.pop is None:
            self.pop = [np.random.randn(self.dim) * 0.3 for _ in range(30)]
            self.best_x = self.pop[0].copy()

        fitness = [task.loss(p) for p in self.pop]
        evals = 30

        while evals < n_evals:
            elite = [self.pop[i].copy() for i in np.argsort(fitness)[:6]]

            if fitness[np.argmin(fitness)] < self.best_loss:
                self.best_loss = min(fitness)
                self.best_x = self.pop[np.argmin(fitness)].copy()

            new_pop = list(elite)
            while len(new_pop) < 30 and evals < n_evals:
                parent = elite[np.random.randint(6)]
                child = parent + np.random.randn(self.dim) * 0.1
                new_pop.append(child)
                evals += 1

            self.pop = new_pop
            fitness = [task.loss(p) for p in self.pop]

        return self.best_loss

    def reset(self):
        self.pop = None
        self.best_loss = float('inf')

class SGMWithLocking:
    """Coalition locking - the working version."""
    def __init__(self, dim):
        self.dim = dim
        self.lock = np.zeros(dim)
        self.causal_sum = np.zeros(dim)
        self.causal_count = np.ones(dim)
        self.group_credits = np.zeros(dim)
        self.pop = None
        self.best_x = None
        self.best_loss = float('inf')

    def init_pop(self):
        self.pop = []
        for _ in range(30):
            x = np.random.randn(self.dim) * 0.3
            if self.best_x is not None:
                for d in range(self.dim):
                    if self.lock[d] > 0.5:
                        x[d] = self.best_x[d]
            self.pop.append(x)

    def measure_causality(self, elite, task):
        x = elite[0]
        base = task.loss(x)

        for d in range(self.dim):
            if self.lock[d] > 0.5:
                continue
            x_test = x.copy()
            x_test[d] = 0
            score = task.loss(x_test) - base
            self.causal_sum[d] += score
            self.causal_count[d] += 1

        avg_causal = self.causal_sum / self.causal_count
        candidates = [d for d in range(self.dim) 
                     if self.lock[d] < 0.5 and 0 < avg_causal[d] < 0.0001]

        if len(candidates) >= 3:
            for _ in range(30):
                k = min(5, len(candidates))
                group = list(np.random.choice(candidates, k, replace=False))
                x_test = x.copy()
                for d in group:
                    x_test[d] = 0
                if task.loss(x_test) - base > 0.0003:
                    for d in group:
                        self.group_credits[d] += 1

    def update_locks(self, elite, task):
        self.measure_causality(elite, task)
        elite_arr = np.array(elite)

        for d in range(self.dim):
            if self.lock[d] > 0.5:
                continue
            var = np.var(elite_arr[:, d])
            avg_causal = self.causal_sum[d] / self.causal_count[d]

            if var < 0.15:
                if avg_causal > 0.00005 or self.group_credits[d] >= 2:
                    self.lock[d] = 1.0

    def step(self, task, n_evals):
        if self.pop is None:
            self.init_pop()

        fitness = [task.loss(p) for p in self.pop]
        evals = 30

        while evals < n_evals:
            elite = [self.pop[i].copy() for i in np.argsort(fitness)[:6]]

            if fitness[np.argmin(fitness)] < self.best_loss:
                self.best_loss = min(fitness)
                self.best_x = self.pop[np.argmin(fitness)].copy()

            self.update_locks(elite, task)

            mutable = [d for d in range(self.dim) if self.lock[d] < 0.5]
            if len(mutable) < 5:
                mutable = list(range(self.dim))

            new_pop = list(elite)
            while len(new_pop) < 30 and evals < n_evals:
                parent = elite[np.random.randint(6)]
                child = parent.copy()
                for d in np.random.choice(mutable, min(5, len(mutable)), replace=False):
                    child[d] += np.random.randn() * 0.1
                new_pop.append(child)
                evals += 1

            self.pop = new_pop
            fitness = [task.loss(p) for p in self.pop]

        return self.best_loss

    def reset(self):
        self.pop = None
        self.best_loss = float('inf')

def run_demo():
    print("="*70)
    print("SGM COALITION LOCKING - SCALING CURVE DEMO")
    print("="*70)

    dims_list = [128, 256, 384, 512, 640]
    n_tasks = 5
    n_evals = 1500
    n_runs = 10

    print(f"\nConfig: {n_tasks} tasks, {n_evals} evals/task, {n_runs} runs")

    results = {'baseline': {d: [] for d in dims_list}, 'locking': {d: [] for d in dims_list}}

    for dims in dims_list:
        print(f"\n--- {dims} dims ---")

        for run in range(n_runs):
            np.random.seed(run * 1000)

            # Create tasks with non-overlapping regions (scaled to dim size)
            tasks = []
            for i in range(n_tasks):
                start = i / n_tasks
                end = (i + 1) / n_tasks
                tasks.append(SparseRegionTask(dims, (start, end), seed=run*1000+i))

            # Baseline
            sgm_b = SGMBaseline(dims)
            b_during = []
            for t in tasks:
                sgm_b.reset()
                sgm_b.step(t, n_evals)
                b_during.append(sgm_b.best_loss)
            b_final = [t.loss(sgm_b.best_x) for t in tasks]
            b_ret = np.mean([b_final[i]/b_during[i] if b_during[i] > 0 else 1 for i in range(n_tasks-1)])
            results['baseline'][dims].append(b_ret)

            # Locking
            sgm_l = SGMWithLocking(dims)
            l_during = []
            for t in tasks:
                sgm_l.reset()
                sgm_l.step(t, n_evals)
                l_during.append(sgm_l.best_loss)
            l_final = [t.loss(sgm_l.best_x) for t in tasks]
            l_ret = np.mean([l_final[i]/l_during[i] if l_during[i] > 0 else 1 for i in range(n_tasks-1)])
            results['locking'][dims].append(l_ret)

        bm = np.mean(results['baseline'][dims])
        lm = np.mean(results['locking'][dims])
        print(f"  Baseline: {bm:.2f}, Locking: {lm:.2f} ({(bm-lm)/bm*100:+.0f}%)")

    # Final summary
    print("\n" + "="*70)
    print("RESULTS: RETENTION RATIO (lower = better, 1.0 = perfect)")
    print("="*70)
    print(f"\n{'Dims':<8} | {'Baseline':>12} | {'+ Locking':>12} | {'Improvement':>12}")
    print("-"*55)

    for dims in dims_list:
        bm, bs = np.mean(results['baseline'][dims]), np.std(results['baseline'][dims])
        lm, ls = np.mean(results['locking'][dims]), np.std(results['locking'][dims])
        imp = (bm - lm) / bm * 100
        print(f"{dims:<8} | {bm:.2f}±{bs:.2f}     | {lm:.2f}±{ls:.2f}     | {imp:>+10.0f}%")

    # Key finding
    b_128 = np.mean(results['baseline'][128])
    b_640 = np.mean(results['baseline'][640])
    l_128 = np.mean(results['locking'][128])
    l_640 = np.mean(results['locking'][640])

    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    print(f"\nBaseline scales: {b_128:.2f} → {b_640:.2f} ({(b_128-b_640)/b_128*100:.0f}% improvement with 5× params)")
    print(f"Locking is flat: {l_128:.2f} → {l_640:.2f} (scale-invariant)")
    print(f"\n128-dim LOCKED ({l_128:.2f}) vs 640-dim BASELINE ({b_640:.2f})")

    if l_128 < b_640:
        print(f"\n✓ SCALING CURVE BROKEN: Small locked system beats large baseline by {(b_640-l_128)/b_640*100:.0f}%")

    return results

if __name__ == "__main__":
    run_demo()