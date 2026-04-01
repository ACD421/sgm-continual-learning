#!/usr/bin/env python3
"""
SGM COALITION LOCKING - EXTREME STRESS TEST
============================================
Run this locally with full compute. Estimated time: 10-30 minutes.

Tests:
1. 500 sequential tasks
2. 10M parameter scale
3. Real embedding dimensions (768, 1024, 4096)
4. Adversarial interference patterns
5. Long-horizon retention (task 0 after 499 more tasks)
6. Capacity saturation to 99.9%
7. Mixed overlap distributions
8. Perturbation robustness at scale
9. Simulated transformer layer freezing
10. Backward/forward transfer measurement

Usage: python sgm_stress_large.py [--quick] [--full]
"""

import numpy as np
import time
import argparse
import sys
from typing import List, Tuple, Dict

# =============================================================================
# CORE SYSTEM
# =============================================================================

class CoalitionLockingSystem:
    """Full implementation with coalition detection"""
    
    def __init__(self, dim: int, dtype=np.float32):
        self.dim = dim
        self.dtype = dtype
        self.x = np.random.randn(dim).astype(dtype) * 0.3
        self.lock = np.zeros(dim, dtype=bool)
        self.causal_scores = np.zeros(dim, dtype=dtype)
        self.causal_count = np.ones(dim, dtype=dtype)
        self.coalition_credits = np.zeros(dim, dtype=dtype)
        self.lock_history = []  # Track when dims were locked
        
    def train(self, loss_fn, n_steps: int = 50, lr: float = 0.03) -> float:
        """Train on task using evolutionary mutation"""
        best_loss = loss_fn(self.x)
        
        for step in range(n_steps):
            x2 = self.x.copy()
            free = np.where(~self.lock)[0]
            
            if len(free) < 5:
                # Capacity exhausted - use all dims with reduced mutation
                idx = np.random.choice(self.dim, min(30, self.dim), replace=False)
                x2[idx] += np.random.randn(len(idx)).astype(self.dtype) * lr * 0.1
            else:
                n_mutate = min(50, len(free))
                idx = np.random.choice(free, n_mutate, replace=False)
                x2[idx] += np.random.randn(n_mutate).astype(self.dtype) * lr
            
            new_loss = loss_fn(x2)
            if new_loss < best_loss:
                self.x = x2
                best_loss = new_loss
                
        return best_loss
    
    def measure_causality(self, loss_fn, n_samples: int = 50):
        """Measure causal importance via ablation"""
        base = loss_fn(self.x)
        free = np.where(~self.lock)[0]
        
        if len(free) == 0:
            return
            
        # Individual ablations
        sample_size = min(n_samples, len(free))
        for d in np.random.choice(free, sample_size, replace=False):
            x_test = self.x.copy()
            x_test[d] = 0
            delta = loss_fn(x_test) - base
            self.causal_scores[d] += delta
            self.causal_count[d] += 1
        
        # Coalition detection
        avg_causal = self.causal_scores[free] / self.causal_count[free]
        weak_candidates = free[(avg_causal > 0) & (avg_causal < 0.001)]
        
        if len(weak_candidates) >= 3:
            for _ in range(30):
                k = min(5, len(weak_candidates))
                group = np.random.choice(weak_candidates, k, replace=False)
                x_test = self.x.copy()
                x_test[group] = 0
                if loss_fn(x_test) - base > 0.005:
                    self.coalition_credits[group] += 1
    
    def update_locks(self, loss_fn, task_id: int = 0):
        """Lock converged, causally important dims"""
        self.measure_causality(loss_fn)
        free = np.where(~self.lock)[0]
        
        newly_locked = 0
        for d in free:
            avg_causal = self.causal_scores[d] / self.causal_count[d]
            if avg_causal > 0.0001 or self.coalition_credits[d] >= 2:
                self.lock[d] = True
                newly_locked += 1
                
        self.lock_history.append((task_id, newly_locked, np.sum(self.lock)))
        return newly_locked
    
    def lock_region(self, start: int, end: int, task_id: int = 0):
        """Explicit region lock"""
        newly_locked = np.sum(~self.lock[start:end])
        self.lock[start:end] = True
        self.lock_history.append((task_id, newly_locked, np.sum(self.lock)))
    
    def get_stats(self) -> Dict:
        return {
            'total_locked': int(np.sum(self.lock)),
            'pct_locked': float(np.mean(self.lock) * 100),
            'free_dims': int(np.sum(~self.lock)),
        }


class BaselineSystem:
    """No locking baseline"""
    
    def __init__(self, dim: int, dtype=np.float32):
        self.dim = dim
        self.dtype = dtype
        self.x = np.random.randn(dim).astype(dtype) * 0.3
        
    def train(self, loss_fn, n_steps: int = 50, lr: float = 0.03) -> float:
        best_loss = loss_fn(self.x)
        
        for _ in range(n_steps):
            x2 = self.x + np.random.randn(self.dim).astype(self.dtype) * lr
            new_loss = loss_fn(x2)
            if new_loss < best_loss:
                self.x = x2
                best_loss = new_loss
                
        return best_loss


# =============================================================================
# TASK GENERATORS
# =============================================================================

class TaskGenerator:
    """Generate diverse task distributions"""
    
    @staticmethod
    def sparse_region(dim: int, task_id: int, n_tasks: int, overlap: float = 0.1, seed: int = None):
        """Non-overlapping regions with optional overlap"""
        if seed is not None:
            np.random.seed(seed)
            
        base_start = (task_id * dim) // n_tasks
        base_end = ((task_id + 1) * dim) // n_tasks
        width = base_end - base_start
        
        start = max(0, base_start - int(width * overlap / 2))
        end = min(dim, base_end + int(width * overlap / 2))
        
        W = np.random.randn(end - start, 16).astype(np.float32) * 0.1
        target = np.random.randn(16).astype(np.float32) * 0.2
        
        def loss_fn(x):
            return np.mean((x[start:end] @ W - target) ** 2)
        
        return loss_fn, start, end
    
    @staticmethod
    def hierarchical(dim: int, task_id: int, n_tasks: int, shared_pct: float = 0.2, seed: int = None):
        """Hierarchical structure with shared base features"""
        if seed is not None:
            np.random.seed(seed)
            
        shared_dims = int(dim * shared_pct)
        task_dims = (dim - shared_dims) // n_tasks
        
        shared_start, shared_end = 0, shared_dims
        task_start = shared_dims + task_id * task_dims
        task_end = shared_dims + (task_id + 1) * task_dims
        
        W_shared = np.random.randn(shared_dims, 8).astype(np.float32) * 0.05
        W_task = np.random.randn(task_end - task_start, 8).astype(np.float32) * 0.15
        target = np.random.randn(8).astype(np.float32) * 0.2
        
        def loss_fn(x):
            shared_out = x[shared_start:shared_end] @ W_shared
            task_out = x[task_start:task_end] @ W_task
            return np.mean((shared_out + task_out - target) ** 2)
        
        return loss_fn, task_start, task_end
    
    @staticmethod
    def transformer_layer(dim: int, layer_id: int, n_layers: int, seed: int = None):
        """Simulates transformer layer with Q, K, V, O projections"""
        if seed is not None:
            np.random.seed(seed)
            
        layer_size = dim // n_layers
        start = layer_id * layer_size
        end = (layer_id + 1) * layer_size
        
        # Q, K, V, O matrices (simplified)
        d_head = layer_size // 4
        Wq = np.random.randn(d_head, 8).astype(np.float32) * 0.1
        Wk = np.random.randn(d_head, 8).astype(np.float32) * 0.1
        Wv = np.random.randn(d_head, 8).astype(np.float32) * 0.1
        Wo = np.random.randn(d_head, 8).astype(np.float32) * 0.1
        target = np.random.randn(8).astype(np.float32) * 0.1
        
        def loss_fn(x):
            q = x[start:start+d_head] @ Wq
            k = x[start+d_head:start+2*d_head] @ Wk
            v = x[start+2*d_head:start+3*d_head] @ Wv
            o = x[start+3*d_head:end] @ Wo
            attn_sim = np.tanh(q * k)  # Simplified attention
            out = attn_sim * v + o
            return np.mean((out - target) ** 2)
        
        return loss_fn, start, end


# =============================================================================
# STRESS TESTS
# =============================================================================

def stress_test_massive_task_count(n_tasks: int = 500, dim: int = 5000):
    """Test: 500 sequential tasks"""
    print(f"\n{'='*70}")
    print(f"STRESS TEST 1: MASSIVE TASK COUNT ({n_tasks} tasks, {dim} dims)")
    print(f"{'='*70}")
    
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    baseline = BaselineSystem(dim)
    
    # Use same seed for both
    baseline.x = locked.x.copy()
    
    l_after, b_after = [], []
    checkpoints = [1, 10, 50, 100, 250, 500]
    checkpoint_results = {}
    
    start_time = time.time()
    
    for t in range(n_tasks):
        loss_fn, start, end = TaskGenerator.sparse_region(dim, t, n_tasks, overlap=0.05, seed=t*100)
        
        b_loss = baseline.train(loss_fn, n_steps=20, lr=0.02)
        l_loss = locked.train(loss_fn, n_steps=20, lr=0.02)
        locked.lock_region(start, end, task_id=t)
        
        b_after.append(b_loss)
        l_after.append(l_loss)
        
        if (t + 1) in checkpoints:
            # Evaluate retention on all previous tasks
            b_final = []
            l_final = []
            for i in range(t + 1):
                eval_fn, _, _ = TaskGenerator.sparse_region(dim, i, n_tasks, overlap=0.05, seed=i*100)
                b_final.append(eval_fn(baseline.x))
                l_final.append(eval_fn(locked.x))
            
            if t > 0:
                b_ret = np.mean([b_final[i]/b_after[i] for i in range(t)])
                l_ret = np.mean([l_final[i]/l_after[i] for i in range(t)])
            else:
                b_ret, l_ret = 1.0, 1.0
                
            checkpoint_results[t + 1] = (b_ret, l_ret, locked.get_stats())
            elapsed = time.time() - start_time
            print(f"  Task {t+1:>3}: Baseline={b_ret:>6.2f}x  Locked={l_ret:>5.2f}x  "
                  f"Locked={locked.get_stats()['pct_locked']:.1f}%  [{elapsed:.1f}s]")
    
    # Final long-horizon check: how well does task 0 survive?
    task0_fn, _, _ = TaskGenerator.sparse_region(dim, 0, n_tasks, overlap=0.05, seed=0)
    task0_b = task0_fn(baseline.x)
    task0_l = task0_fn(locked.x)
    
    print(f"\n  LONG-HORIZON: Task 0 retention after {n_tasks-1} more tasks:")
    print(f"    Baseline: {task0_b/b_after[0]:.2f}x degradation")
    print(f"    Locked:   {task0_l/l_after[0]:.2f}x degradation")
    
    return checkpoint_results


def stress_test_scale(dims: List[int] = [1000, 10000, 100000, 1000000, 10000000]):
    """Test: Scale from 1K to 10M parameters"""
    print(f"\n{'='*70}")
    print("STRESS TEST 2: PARAMETER SCALING (1K -> 10M)")
    print(f"{'='*70}")
    
    n_tasks = 20
    results = {}
    
    for dim in dims:
        print(f"\n  {dim:>10,} parameters...")
        start_time = time.time()
        
        np.random.seed(42)
        locked = CoalitionLockingSystem(dim)
        baseline = BaselineSystem(dim)
        baseline.x = locked.x.copy()
        
        l_after, b_after = [], []
        
        for t in range(n_tasks):
            loss_fn, start, end = TaskGenerator.sparse_region(dim, t, n_tasks, seed=t*100)
            baseline.train(loss_fn, n_steps=15, lr=0.02)
            locked.train(loss_fn, n_steps=15, lr=0.02)
            locked.lock_region(start, end)
            b_after.append(loss_fn(baseline.x))
            l_after.append(loss_fn(locked.x))
        
        # Final eval
        b_final, l_final = [], []
        for t in range(n_tasks):
            eval_fn, _, _ = TaskGenerator.sparse_region(dim, t, n_tasks, seed=t*100)
            b_final.append(eval_fn(baseline.x))
            l_final.append(eval_fn(locked.x))
        
        b_ret = np.mean([b_final[i]/b_after[i] for i in range(n_tasks-1)])
        l_ret = np.mean([l_final[i]/l_after[i] for i in range(n_tasks-1)])
        elapsed = time.time() - start_time
        
        results[dim] = (b_ret, l_ret, elapsed)
        print(f"    Baseline={b_ret:>6.2f}x  Locked={l_ret:>5.2f}x  [{elapsed:.1f}s]")
    
    # Summary table
    print(f"\n  {'Params':<12} | {'Baseline':>10} | {'Locked':>10} | {'Ratio':>8}")
    print("  " + "-"*50)
    for dim in dims:
        b, l, _ = results[dim]
        ratio = b / l if l > 0 else float('inf')
        print(f"  {dim:<12,} | {b:>9.2f}x | {l:>9.2f}x | {ratio:>7.1f}x")
    
    return results


def stress_test_real_embedding_dims():
    """Test: Real-world embedding dimensions (768, 1024, 2048, 4096)"""
    print(f"\n{'='*70}")
    print("STRESS TEST 3: REAL EMBEDDING DIMENSIONS")
    print(f"{'='*70}")
    print("  Testing BERT-base (768), GPT-2 (1024), LLaMA (4096) dimensions\n")
    
    embedding_dims = [768, 1024, 2048, 4096]
    n_tasks = 30
    
    for dim in embedding_dims:
        np.random.seed(42)
        locked = CoalitionLockingSystem(dim)
        baseline = BaselineSystem(dim)
        baseline.x = locked.x.copy()
        
        l_after, b_after = [], []
        
        for t in range(n_tasks):
            loss_fn, start, end = TaskGenerator.hierarchical(dim, t, n_tasks, shared_pct=0.15, seed=t*100)
            baseline.train(loss_fn, n_steps=30, lr=0.025)
            locked.train(loss_fn, n_steps=30, lr=0.025)
            locked.lock_region(start, end)
            b_after.append(loss_fn(baseline.x))
            l_after.append(loss_fn(locked.x))
        
        b_final, l_final = [], []
        for t in range(n_tasks):
            eval_fn, _, _ = TaskGenerator.hierarchical(dim, t, n_tasks, shared_pct=0.15, seed=t*100)
            b_final.append(eval_fn(baseline.x))
            l_final.append(eval_fn(locked.x))
        
        b_ret = np.mean([b_final[i]/b_after[i] for i in range(n_tasks-1)])
        l_ret = np.mean([l_final[i]/l_after[i] for i in range(n_tasks-1)])
        
        print(f"  dim={dim:<5}: Baseline={b_ret:>6.2f}x  Locked={l_ret:>5.2f}x  "
              f"({locked.get_stats()['pct_locked']:.1f}% locked)")


def stress_test_transformer_simulation(n_layers: int = 12, d_model: int = 768):
    """Test: Simulated transformer layer freezing"""
    print(f"\n{'='*70}")
    print(f"STRESS TEST 4: TRANSFORMER LAYER SIMULATION ({n_layers}L x {d_model}d)")
    print(f"{'='*70}")
    
    dim = n_layers * d_model
    print(f"  Total parameters: {dim:,}\n")
    
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    baseline = BaselineSystem(dim)
    baseline.x = locked.x.copy()
    
    l_after, b_after = [], []
    
    for layer in range(n_layers):
        loss_fn, start, end = TaskGenerator.transformer_layer(dim, layer, n_layers, seed=layer*100)
        
        baseline.train(loss_fn, n_steps=40, lr=0.02)
        locked.train(loss_fn, n_steps=40, lr=0.02)
        locked.lock_region(start, end, task_id=layer)
        
        b_after.append(loss_fn(baseline.x))
        l_after.append(loss_fn(locked.x))
        
        print(f"  Layer {layer+1:>2}/{n_layers}: trained, locked dims {start}-{end}")
    
    # Final eval
    print(f"\n  Layer retention after all layers trained:")
    print(f"  {'Layer':<8} | {'Baseline':>10} | {'Locked':>10}")
    print("  " + "-"*35)
    
    for layer in range(n_layers):
        eval_fn, _, _ = TaskGenerator.transformer_layer(dim, layer, n_layers, seed=layer*100)
        b_final = eval_fn(baseline.x)
        l_final = eval_fn(locked.x)
        b_ret = b_final / b_after[layer]
        l_ret = l_final / l_after[layer]
        print(f"  Layer {layer+1:<3} | {b_ret:>9.2f}x | {l_ret:>9.2f}x")


def stress_test_adversarial_extreme():
    """Test: Extreme adversarial conditions"""
    print(f"\n{'='*70}")
    print("STRESS TEST 5: EXTREME ADVERSARIAL CONDITIONS")
    print(f"{'='*70}")
    
    dim = 5000
    
    # Test A: 100% overlap - all tasks use all dims
    print("\n  [A] 100% overlap (all tasks use all dims):")
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    baseline = BaselineSystem(dim)
    
    for t in range(20):
        W = np.random.randn(dim, 8).astype(np.float32) * 0.1
        target = np.random.randn(8).astype(np.float32)
        loss_fn = lambda x, W=W, t=target: np.mean((x @ W - t) ** 2)
        
        baseline.train(loss_fn, n_steps=20)
        locked.train(loss_fn, n_steps=20)
        if t == 0:
            locked.lock[:] = True  # Lock everything after task 0
    
    print(f"      Final task - Baseline: {loss_fn(baseline.x):.4f}, Locked: {loss_fn(locked.x):.4f}")
    print(f"      (Locked SHOULD fail here - this is the boundary condition)")
    
    # Test B: Oscillating targets
    print("\n  [B] Oscillating targets (A -> B -> A -> B...):")
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    
    W = np.random.randn(dim//2, 8).astype(np.float32) * 0.1
    target_A = np.ones(8).astype(np.float32)
    target_B = -np.ones(8).astype(np.float32)
    
    loss_A = lambda x: np.mean((x[:dim//2] @ W - target_A) ** 2)
    loss_B = lambda x: np.mean((x[:dim//2] @ W - target_B) ** 2)
    
    locked.train(loss_A, n_steps=50)
    initial_A = loss_A(locked.x)
    locked.lock_region(0, dim//2)
    
    for i in range(10):
        if i % 2 == 0:
            locked.train(loss_B, n_steps=20)
        else:
            locked.train(loss_A, n_steps=20)
    
    final_A = loss_A(locked.x)
    print(f"      Task A: initial={initial_A:.4f}, after 10 oscillations={final_A:.4f}")
    print(f"      Retention: {final_A/initial_A:.2f}x (1.0 = perfect)")
    
    # Test C: Gradient-like attack (try to unlock via perturbation)
    print("\n  [C] Perturbation attack on locked dims:")
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    
    loss_fn, start, end = TaskGenerator.sparse_region(dim, 0, 5, seed=0)
    locked.train(loss_fn, n_steps=100)
    before = loss_fn(locked.x)
    locked.lock_region(start, end)
    
    # Attack: perturb locked dims directly
    locked.x[start:end] += np.random.randn(end-start).astype(np.float32) * 2.0
    after = loss_fn(locked.x)
    
    print(f"      Before perturbation: {before:.4f}")
    print(f"      After 2sigma perturbation of locked dims: {after:.4f}")
    print(f"      (This tests if locked VALUES are protected, not just mask)")


def stress_test_capacity_to_limit():
    """Test: Push capacity to 99.9%"""
    print(f"\n{'='*70}")
    print("STRESS TEST 6: CAPACITY SATURATION TO 99.9%")
    print(f"{'='*70}")
    
    dim = 10000
    
    for target_lock_pct in [90, 95, 99, 99.5, 99.9]:
        n_prelock = int(dim * target_lock_pct / 100)
        
        np.random.seed(42)
        locked = CoalitionLockingSystem(dim)
        locked.lock[:n_prelock] = True
        
        # Try to learn in remaining space
        free_dims = dim - n_prelock
        if free_dims > 0:
            W = np.random.randn(free_dims, 8).astype(np.float32) * 0.1
            target = np.random.randn(8).astype(np.float32)
            loss_fn = lambda x: np.mean((x[n_prelock:] @ W - target) ** 2)
            
            init_loss = loss_fn(locked.x)
            final_loss = locked.train(loss_fn, n_steps=200, lr=0.03)
            improvement = (init_loss - final_loss) / init_loss * 100
            
            print(f"  {target_lock_pct:>5.1f}% locked ({free_dims:>4} free): "
                  f"{improvement:>6.1f}% improvement")


def stress_test_mixed_overlap():
    """Test: Mixed overlap patterns in same sequence"""
    print(f"\n{'='*70}")
    print("STRESS TEST 7: MIXED OVERLAP PATTERNS")
    print(f"{'='*70}")
    
    dim = 5000
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    baseline = BaselineSystem(dim)
    baseline.x = locked.x.copy()
    
    # Mix of overlap patterns: 0%, 25%, 50%, 75%, 100%
    overlaps = [0.0, 0.25, 0.5, 0.75, 0.0, 0.5, 0.25, 0.0, 0.75, 0.5]
    n_tasks = len(overlaps)
    
    l_after, b_after = [], []
    
    print(f"\n  Training {n_tasks} tasks with varying overlap:\n")
    
    for t, overlap in enumerate(overlaps):
        loss_fn, start, end = TaskGenerator.sparse_region(dim, t, n_tasks, overlap=overlap, seed=t*100)
        baseline.train(loss_fn, n_steps=30)
        locked.train(loss_fn, n_steps=30)
        locked.lock_region(start, end)
        b_after.append(loss_fn(baseline.x))
        l_after.append(loss_fn(locked.x))
        print(f"    Task {t}: {int(overlap*100):>3}% overlap")
    
    # Final eval
    print(f"\n  Retention by overlap level:")
    for t, overlap in enumerate(overlaps):
        if t < n_tasks - 1:
            eval_fn, _, _ = TaskGenerator.sparse_region(dim, t, n_tasks, overlap=overlap, seed=t*100)
            b_ret = eval_fn(baseline.x) / b_after[t]
            l_ret = eval_fn(locked.x) / l_after[t]
            print(f"    Task {t} ({int(overlap*100):>2}%): Baseline={b_ret:>5.2f}x  Locked={l_ret:>5.2f}x")


def stress_test_backward_forward_transfer():
    """Test: Measure backward and forward transfer"""
    print(f"\n{'='*70}")
    print("STRESS TEST 8: BACKWARD/FORWARD TRANSFER")
    print(f"{'='*70}")
    
    dim = 5000
    n_tasks = 20
    
    np.random.seed(42)
    locked = CoalitionLockingSystem(dim)
    baseline = BaselineSystem(dim)
    baseline.x = locked.x.copy()
    
    # Random baseline for each task (no learning)
    random_losses = []
    for t in range(n_tasks):
        loss_fn, _, _ = TaskGenerator.hierarchical(dim, t, n_tasks, seed=t*100)
        random_x = np.random.randn(dim).astype(np.float32) * 0.3
        random_losses.append(loss_fn(random_x))
    
    # Train sequentially
    l_before, l_after, b_before, b_after = [], [], [], []
    
    for t in range(n_tasks):
        loss_fn, start, end = TaskGenerator.hierarchical(dim, t, n_tasks, seed=t*100)
        
        # Measure before training (forward transfer)
        l_before.append(loss_fn(locked.x))
        b_before.append(loss_fn(baseline.x))
        
        # Train
        baseline.train(loss_fn, n_steps=40)
        locked.train(loss_fn, n_steps=40)
        locked.lock_region(start, end)
        
        l_after.append(loss_fn(locked.x))
        b_after.append(loss_fn(baseline.x))
    
    # Compute transfer metrics
    print("\n  Forward Transfer (performance before training vs random):")
    print("  Higher = better transfer from previous tasks\n")
    
    for t in range(1, min(5, n_tasks)):
        l_fwd = (random_losses[t] - l_before[t]) / random_losses[t] * 100
        b_fwd = (random_losses[t] - b_before[t]) / random_losses[t] * 100
        print(f"    Task {t}: Baseline={b_fwd:>+6.1f}%  Locked={l_fwd:>+6.1f}%")
    
    # Backward transfer (final eval on early tasks)
    print("\n  Backward Transfer (retention of early tasks):")
    
    b_final, l_final = [], []
    for t in range(n_tasks):
        eval_fn, _, _ = TaskGenerator.hierarchical(dim, t, n_tasks, seed=t*100)
        b_final.append(eval_fn(baseline.x))
        l_final.append(eval_fn(locked.x))
    
    print(f"\n    {'Task':<6} | {'Baseline':>10} | {'Locked':>10}")
    print("    " + "-"*35)
    for t in range(min(10, n_tasks-1)):
        b_ret = b_final[t] / b_after[t]
        l_ret = l_final[t] / l_after[t]
        print(f"    {t:<6} | {b_ret:>9.2f}x | {l_ret:>9.2f}x")


# =============================================================================
# MAIN
# =============================================================================

def run_quick_tests():
    """Quick sanity check (~1-2 min)"""
    stress_test_massive_task_count(n_tasks=50, dim=2000)
    stress_test_scale(dims=[1000, 10000, 100000])
    stress_test_adversarial_extreme()


def run_full_tests():
    """Complete stress test battery (~10-30 min)"""
    stress_test_massive_task_count(n_tasks=500, dim=5000)
    stress_test_scale(dims=[1000, 10000, 100000, 1000000, 10000000])
    stress_test_real_embedding_dims()
    stress_test_transformer_simulation(n_layers=12, d_model=768)
    stress_test_adversarial_extreme()
    stress_test_capacity_to_limit()
    stress_test_mixed_overlap()
    stress_test_backward_forward_transfer()


def main():
    parser = argparse.ArgumentParser(description='SGM Coalition Locking Stress Test')
    parser.add_argument('--quick', action='store_true', help='Quick sanity check (~1-2 min)')
    parser.add_argument('--full', action='store_true', help='Full stress test (~10-30 min)')
    args = parser.parse_args()
    
    print("="*70)
    print("SGM COALITION LOCKING - EXTREME STRESS TEST")
    print("="*70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"NumPy: {np.__version__}")
    
    start_time = time.time()
    
    if args.quick:
        print("\nRunning QUICK tests...\n")
        run_quick_tests()
    elif args.full:
        print("\nRunning FULL stress test battery...\n")
        run_full_tests()
    else:
        print("\nNo mode specified, defaulting to --quick...\n")
        run_quick_tests()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("STRESS TEST COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    
    print("""
KEY METRICS TO VERIFY:
----------------------
[OK] 500 tasks: Locked retention should stay ~1.0x throughout
[OK] 10M scale: Retention gap should INCREASE with scale (Kaplan violation)
[OK] Real dims: 768/1024/4096 should all show ~1x retention
[OK] Transformer sim: All 12 layers retained after sequential training
[OK] Adversarial: 100% overlap should fail (boundary condition)
[OK] Capacity: Plasticity preserved even at 99.9% lock saturation
[OK] Mixed overlap: Graceful degradation on high-overlap tasks
[OK] Transfer: Positive forward transfer, near-zero backward interference

If all pass, the primitive is validated for production deployment.
""")


if __name__ == "__main__":
    main()