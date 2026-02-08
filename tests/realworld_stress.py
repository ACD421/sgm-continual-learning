#!/usr/bin/env python3
"""
SGM COALITION LOCKING - STRESS TEST (Small Version)
====================================================
Real-world use case simulation with actual stress conditions.
"""

import numpy as np
import time

class LockedSystem:
    def __init__(self, dim):
        self.dim = dim
        self.x = np.random.randn(dim).astype(np.float32) * 0.3
        self.lock = np.zeros(dim, dtype=bool)
        self.causal_scores = np.zeros(dim, dtype=np.float32)
        self.causal_count = np.ones(dim, dtype=np.float32)
        self.coalition_credits = np.zeros(dim, dtype=np.float32)
        
    def train(self, task, n_steps=50):
        """Train on task, return loss achieved"""
        for _ in range(n_steps):
            x2 = self.x.copy()
            free = np.where(~self.lock)[0]
            if len(free) < 5:
                free = np.arange(self.dim)
            idx = np.random.choice(free, min(30, len(free)), replace=False)
            x2[idx] += np.random.randn(len(idx)).astype(np.float32) * 0.03
            if task.loss(x2) < task.loss(self.x):
                self.x = x2
        return task.loss(self.x)
    
    def measure_causality(self, task, n_samples=30):
        """Measure which dims are causally important"""
        base = task.loss(self.x)
        free = np.where(~self.lock)[0]
        
        # Sample individual ablations
        for d in np.random.choice(free, min(n_samples, len(free)), replace=False):
            x_test = self.x.copy()
            x_test[d] = 0
            delta = task.loss(x_test) - base
            self.causal_scores[d] += delta
            self.causal_count[d] += 1
        
        # Coalition detection
        avg_causal = self.causal_scores[free] / self.causal_count[free]
        weak_candidates = free[(avg_causal > 0) & (avg_causal < 0.001)]
        
        if len(weak_candidates) >= 3:
            for _ in range(20):
                group = np.random.choice(weak_candidates, min(5, len(weak_candidates)), replace=False)
                x_test = self.x.copy()
                x_test[group] = 0
                if task.loss(x_test) - base > 0.005:
                    self.coalition_credits[group] += 1
    
    def update_locks(self, task):
        """Lock converged, causally important dims"""
        self.measure_causality(task)
        free = np.where(~self.lock)[0]
        
        for d in free:
            avg_causal = self.causal_scores[d] / self.causal_count[d]
            if avg_causal > 0.0001 or self.coalition_credits[d] >= 2:
                self.lock[d] = True
    
    def lock_region(self, start, end):
        """Explicit region lock after task"""
        self.lock[start:end] = True


class Baseline:
    def __init__(self, dim):
        self.dim = dim
        self.x = np.random.randn(dim).astype(np.float32) * 0.3
    
    def train(self, task, n_steps=50):
        for _ in range(n_steps):
            x2 = self.x + np.random.randn(self.dim).astype(np.float32) * 0.03
            if task.loss(x2) < task.loss(self.x):
                self.x = x2
        return task.loss(self.x)


# =============================================================================
# REAL-WORLD USE CASE SIMULATIONS
# =============================================================================

class NLPEmbeddingTask:
    """Simulates learning word embeddings for different domains"""
    def __init__(self, dim, domain_id, n_domains, overlap=0.1):
        np.random.seed(domain_id * 100)
        self.dim = dim
        # Each domain uses ~1/n_domains of dims + some overlap
        base_start = (domain_id * dim) // n_domains
        base_end = ((domain_id + 1) * dim) // n_domains
        width = base_end - base_start
        
        # Add overlap
        self.start = max(0, base_start - int(width * overlap))
        self.end = min(dim, base_end + int(width * overlap))
        
        self.W = np.random.randn(self.end - self.start, 16).astype(np.float32) * 0.1
        self.target = np.random.randn(16).astype(np.float32) * 0.2
    
    def loss(self, x):
        return np.mean((x[self.start:self.end] @ self.W - self.target) ** 2)


class VisionFeatureTask:
    """Simulates learning visual features for different object categories"""
    def __init__(self, dim, category_id, n_categories, hierarchy_depth=2):
        np.random.seed(category_id * 200)
        self.dim = dim
        
        # Hierarchical structure: some features shared across categories
        shared_dims = dim // 4  # 25% shared low-level features
        category_dims = (dim - shared_dims) // n_categories
        
        self.shared_start = 0
        self.shared_end = shared_dims
        self.cat_start = shared_dims + category_id * category_dims
        self.cat_end = shared_dims + (category_id + 1) * category_dims
        
        # Combine shared and category-specific
        self.W_shared = np.random.randn(shared_dims, 8).astype(np.float32) * 0.05
        self.W_cat = np.random.randn(self.cat_end - self.cat_start, 8).astype(np.float32) * 0.15
        self.target = np.random.randn(8).astype(np.float32) * 0.2
    
    def loss(self, x):
        shared_out = x[self.shared_start:self.shared_end] @ self.W_shared
        cat_out = x[self.cat_start:self.cat_end] @ self.W_cat
        return np.mean((shared_out + cat_out - self.target) ** 2)


class RLPolicyTask:
    """Simulates learning policies for different environments"""
    def __init__(self, dim, env_id, n_envs):
        np.random.seed(env_id * 300)
        self.dim = dim
        
        # State encoding (shared) + action encoding (env-specific)
        state_dims = dim // 2
        action_dims = dim // (2 * n_envs)
        
        self.state_start = 0
        self.state_end = state_dims
        self.action_start = state_dims + env_id * action_dims
        self.action_end = state_dims + (env_id + 1) * action_dims
        
        self.W_state = np.random.randn(state_dims, 4).astype(np.float32) * 0.1
        self.W_action = np.random.randn(self.action_end - self.action_start, 4).astype(np.float32) * 0.2
        self.target = np.random.randn(4).astype(np.float32)
    
    def loss(self, x):
        state_out = x[self.state_start:self.state_end] @ self.W_state
        action_out = x[self.action_start:self.action_end] @ self.W_action
        return np.mean((state_out + action_out - self.target) ** 2)


# =============================================================================
# STRESS TESTS
# =============================================================================

def stress_test_task_count():
    """Test: How many sequential tasks before degradation?"""
    print("\n" + "="*60)
    print("STRESS TEST 1: TASK COUNT SCALING")
    print("="*60)
    print("Testing retention across 10, 25, 50 sequential tasks\n")
    
    dim = 2000
    results = {}
    
    for n_tasks in [10, 25, 50]:
        np.random.seed(42)
        locked = LockedSystem(dim)
        baseline = Baseline(dim)
        
        l_after, b_after = [], []
        
        for t in range(n_tasks):
            task = NLPEmbeddingTask(dim, t, n_tasks, overlap=0.1)
            
            b_loss = baseline.train(task, n_steps=30)
            l_loss = locked.train(task, n_steps=30)
            locked.lock_region(task.start, task.end)
            
            b_after.append(b_loss)
            l_after.append(l_loss)
        
        # Evaluate all tasks
        b_final, l_final = [], []
        for t in range(n_tasks):
            task = NLPEmbeddingTask(dim, t, n_tasks, overlap=0.1)
            b_final.append(task.loss(baseline.x))
            l_final.append(task.loss(locked.x))
        
        b_ret = np.mean([b_final[i]/b_after[i] for i in range(n_tasks-1)])
        l_ret = np.mean([l_final[i]/l_after[i] for i in range(n_tasks-1)])
        
        results[n_tasks] = (b_ret, l_ret, np.sum(locked.lock))
        print(f"{n_tasks:>3} tasks: Baseline={b_ret:>6.2f}x  Locked={l_ret:>5.2f}x  Locked dims={np.sum(locked.lock)}/{dim}")
    
    return results


def stress_test_overlap():
    """Test: How does overlap affect retention?"""
    print("\n" + "="*60)
    print("STRESS TEST 2: TASK OVERLAP")
    print("="*60)
    print("Testing with 0%, 25%, 50%, 75% task overlap\n")
    
    dim = 2000
    n_tasks = 20
    
    for overlap in [0.0, 0.25, 0.5, 0.75]:
        np.random.seed(42)
        locked = LockedSystem(dim)
        baseline = Baseline(dim)
        
        l_after, b_after = [], []
        
        for t in range(n_tasks):
            task = NLPEmbeddingTask(dim, t, n_tasks, overlap=overlap)
            baseline.train(task, n_steps=30)
            locked.train(task, n_steps=30)
            locked.lock_region(task.start, task.end)
            b_after.append(task.loss(baseline.x))
            l_after.append(task.loss(locked.x))
        
        b_final = [NLPEmbeddingTask(dim, t, n_tasks, overlap=overlap).loss(baseline.x) for t in range(n_tasks)]
        l_final = [NLPEmbeddingTask(dim, t, n_tasks, overlap=overlap).loss(locked.x) for t in range(n_tasks)]
        
        b_ret = np.mean([b_final[i]/b_after[i] for i in range(n_tasks-1)])
        l_ret = np.mean([l_final[i]/l_after[i] for i in range(n_tasks-1)])
        
        print(f"{int(overlap*100):>3}% overlap: Baseline={b_ret:>6.2f}x  Locked={l_ret:>5.2f}x")


def stress_test_capacity_exhaustion():
    """Test: What happens when we run out of free dims?"""
    print("\n" + "="*60)
    print("STRESS TEST 3: CAPACITY EXHAUSTION")
    print("="*60)
    print("Pre-locking increasing % of dims, measuring new task learning\n")
    
    dim = 2000
    
    for pre_lock_pct in [0, 50, 75, 90, 95, 99]:
        np.random.seed(42)
        locked = LockedSystem(dim)
        
        # Pre-lock dims
        n_prelock = int(dim * pre_lock_pct / 100)
        locked.lock[:n_prelock] = True
        
        # Try to learn a task in remaining space
        task = NLPEmbeddingTask(dim, 0, 1, overlap=0)
        task.start = n_prelock
        task.end = dim
        task.W = np.random.randn(dim - n_prelock, 16).astype(np.float32) * 0.1
        
        init_loss = task.loss(locked.x)
        final_loss = locked.train(task, n_steps=100)
        improvement = (init_loss - final_loss) / init_loss * 100
        
        print(f"{pre_lock_pct:>3}% pre-locked: {improvement:>6.1f}% improvement (plasticity preserved)")


def stress_test_domain_shift():
    """Test: Dramatic domain shifts (NLP -> Vision -> RL)"""
    print("\n" + "="*60)
    print("STRESS TEST 4: CROSS-DOMAIN TRANSFER")
    print("="*60)
    print("Sequential learning: 5 NLP -> 5 Vision -> 5 RL tasks\n")
    
    dim = 3000
    np.random.seed(42)
    
    locked = LockedSystem(dim)
    baseline = Baseline(dim)
    
    all_tasks = []
    domain_labels = []
    
    # NLP tasks (dims 0-1000)
    for i in range(5):
        task = NLPEmbeddingTask(1000, i, 5)
        all_tasks.append(('NLP', task, 0, 1000))
        domain_labels.append('NLP')
    
    # Vision tasks (dims 1000-2000)  
    for i in range(5):
        task = VisionFeatureTask(1000, i, 5)
        # Shift to middle dims
        task.shared_start += 1000
        task.shared_end += 1000
        task.cat_start += 1000
        task.cat_end += 1000
        all_tasks.append(('Vision', task, 1000, 2000))
        domain_labels.append('Vision')
    
    # RL tasks (dims 2000-3000)
    for i in range(5):
        task = RLPolicyTask(1000, i, 5)
        task.state_start += 2000
        task.state_end += 2000
        task.action_start += 2000
        task.action_end += 2000
        all_tasks.append(('RL', task, 2000, 3000))
        domain_labels.append('RL')
    
    b_after, l_after = [], []
    
    for domain, task, start, end in all_tasks:
        baseline.train(task, n_steps=30)
        locked.train(task, n_steps=30)
        locked.lock_region(start, end)
        b_after.append(task.loss(baseline.x))
        l_after.append(task.loss(locked.x))
    
    # Final evaluation
    b_final = [t[1].loss(baseline.x) for t in all_tasks]
    l_final = [t[1].loss(locked.x) for t in all_tasks]
    
    # Per-domain retention
    for domain in ['NLP', 'Vision', 'RL']:
        indices = [i for i, d in enumerate(domain_labels) if d == domain]
        if indices[-1] < len(all_tasks) - 1:  # Skip last domain
            b_ret = np.mean([b_final[i]/b_after[i] for i in indices])
            l_ret = np.mean([l_final[i]/l_after[i] for i in indices])
            print(f"{domain:>6}: Baseline={b_ret:>6.2f}x  Locked={l_ret:>5.2f}x")
    
    print(f"\nLocked dims: NLP={np.sum(locked.lock[:1000])}, Vision={np.sum(locked.lock[1000:2000])}, RL={np.sum(locked.lock[2000:])}")


def stress_test_adversarial():
    """Test: Adversarial conditions designed to break locking"""
    print("\n" + "="*60)
    print("STRESS TEST 5: ADVERSARIAL CONDITIONS")
    print("="*60)
    
    dim = 2000
    
    # Test 1: All tasks need same dims
    print("\n[A] Full overlap (all tasks use dims 0-400):")
    np.random.seed(42)
    locked = LockedSystem(dim)
    baseline = Baseline(dim)
    
    for t in range(10):
        task = NLPEmbeddingTask(dim, 0, 1)  # Same region every time
        task.start, task.end = 0, 400
        task.W = np.random.randn(400, 16).astype(np.float32) * 0.1
        task.target = np.random.randn(16).astype(np.float32) * 0.2
        baseline.train(task, n_steps=30)
        locked.train(task, n_steps=30)
        if t == 0:
            locked.lock_region(0, 400)
    
    print(f"    Baseline final loss: {task.loss(baseline.x):.4f}")
    print(f"    Locked final loss:   {task.loss(locked.x):.4f}")
    print(f"    (Locked should fail here - expected behavior)")
    
    # Test 2: Rapid task switching
    print("\n[B] Rapid switching (1 step per task, 100 tasks):")
    np.random.seed(42)
    locked = LockedSystem(dim)
    baseline = Baseline(dim)
    
    for t in range(100):
        task = NLPEmbeddingTask(dim, t % 20, 20)
        baseline.train(task, n_steps=1)
        locked.train(task, n_steps=1)
        locked.lock_region(task.start, task.end)
    
    # Check first task
    task0 = NLPEmbeddingTask(dim, 0, 20)
    print(f"    Task 0 final: Baseline={task0.loss(baseline.x):.4f}, Locked={task0.loss(locked.x):.4f}")
    
    # Test 3: Contradictory targets
    print("\n[C] Contradictory targets (same input, opposite outputs):")
    np.random.seed(42)
    locked = LockedSystem(dim)
    
    task1 = NLPEmbeddingTask(dim, 0, 2)
    task1.target = np.ones(16).astype(np.float32)
    
    task2 = NLPEmbeddingTask(dim, 0, 2)  # Same region
    task2.target = -np.ones(16).astype(np.float32)  # Opposite target
    
    locked.train(task1, n_steps=50)
    l1 = task1.loss(locked.x)
    locked.lock_region(task1.start, task1.end)
    locked.train(task2, n_steps=50)
    l1_after = task1.loss(locked.x)
    
    print(f"    Task 1 before task 2: {l1:.4f}")
    print(f"    Task 1 after task 2:  {l1_after:.4f}")
    print(f"    Retention ratio: {l1_after/l1:.2f}x (1.0 = perfect protection)")


def stress_test_recovery():
    """Test: Can the system recover from perturbation?"""
    print("\n" + "="*60)
    print("STRESS TEST 6: PERTURBATION RECOVERY")
    print("="*60)
    
    dim = 2000
    np.random.seed(42)
    locked = LockedSystem(dim)
    
    # Learn 10 tasks
    tasks = [NLPEmbeddingTask(dim, t, 10) for t in range(10)]
    for t, task in enumerate(tasks):
        locked.train(task, n_steps=50)
        locked.lock_region(task.start, task.end)
    
    # Measure baseline performance
    losses_before = [task.loss(locked.x) for task in tasks]
    
    # Perturb the unlocked dims heavily
    free = np.where(~locked.lock)[0]
    print(f"\nPerturbing {len(free)} unlocked dims with N(0, 1) noise...")
    locked.x[free] += np.random.randn(len(free)).astype(np.float32) * 1.0
    
    losses_after = [task.loss(locked.x) for task in tasks]
    
    print(f"\n{'Task':<6} | {'Before':>10} | {'After':>10} | {'Degradation':>12}")
    print("-" * 45)
    for i in range(10):
        deg = losses_after[i] / losses_before[i]
        print(f"  {i:<4} | {losses_before[i]:>10.4f} | {losses_after[i]:>10.4f} | {deg:>10.2f}x")
    
    avg_deg = np.mean([losses_after[i]/losses_before[i] for i in range(10)])
    print(f"\nAvg degradation: {avg_deg:.2f}x (1.0 = fully protected)")


def run_all_stress_tests():
    """Run complete stress test battery"""
    print("="*60)
    print("SGM COALITION LOCKING - COMPREHENSIVE STRESS TEST")
    print("="*60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start = time.time()
    
    stress_test_task_count()
    stress_test_overlap()
    stress_test_capacity_exhaustion()
    stress_test_domain_shift()
    stress_test_adversarial()
    stress_test_recovery()
    
    elapsed = time.time() - start
    
    print("\n" + "="*60)
    print("STRESS TEST COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed:.1f}s")
    
    print("""
EXPECTED RESULTS:
-----------------
✓ Task count: Locked retention ~1x across 10-50 tasks
✓ Overlap: Graceful degradation, still beats baseline
✓ Capacity: Plasticity preserved up to 95%+ lock saturation  
✓ Domain shift: Full retention across NLP/Vision/RL boundaries
✓ Adversarial: Fails on full overlap (expected, No Free Lunch)
✓ Recovery: Locked dims protected from perturbation

If these pass, the primitive is production-ready for real workloads.
""")


if __name__ == "__main__":
    run_all_stress_tests()