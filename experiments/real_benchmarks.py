#!/usr/bin/env python3
"""
SGM REAL BENCHMARK SUITE - COMPLETE
====================================
Run this on your machine with: python SGM_REAL_BENCHMARKS_FINAL.py

Requirements:
    pip install torch torchvision numpy

Benchmarks:
    1. Split-MNIST (5 tasks)
    2. Permuted-MNIST (10 permutations)
    3. Split-CIFAR-100 (10 tasks)

Methods Compared:
    1. Baseline (naive sequential)
    2. EWC (Elastic Weight Consolidation - Kirkpatrick 2017)
    3. SGM (Ours - Stochastic Gradient Mutation with Locking)

Output:
    - Accuracy matrices
    - Final accuracy (mean +/- std over 3 seeds)
    - Backward transfer (forgetting metric)
    - Statistical comparison

Expected runtime: ~20-30 minutes on CPU, ~5-10 minutes on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS = 3  # For error bars
DATA_DIR = './data'

print("="*70)
print("SGM REAL BENCHMARK SUITE")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Seeds: {N_SEEDS}")
print("="*70)

# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class MLP(nn.Module):
    """MLP for MNIST (784 -> 256 -> 256 -> 10)"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    """CNN for CIFAR-100"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train, test


def get_cifar100():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=transform_test)
    return train, test


def make_split_mnist_tasks(train_data, test_data, n_tasks=5):
    """Split-MNIST: 0/1, 2/3, 4/5, 6/7, 8/9"""
    tasks = []
    classes_per_task = 10 // n_tasks
    
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = [i for i, (_, y) in enumerate(train_data) if y in task_classes]
        test_idx = [i for i, (_, y) in enumerate(test_data) if y in task_classes]
        
        tasks.append({
            'train': Subset(train_data, train_idx),
            'test': Subset(test_data, test_idx),
            'classes': task_classes,
            'name': f"Digits {task_classes}"
        })
    return tasks


def make_permuted_mnist_tasks(train_data, test_data, n_tasks=10):
    """Permuted-MNIST: same digits, different pixel orderings"""
    tasks = []
    for t in range(n_tasks):
        np.random.seed(t * 1000)
        perm = np.random.permutation(784)
        tasks.append({
            'train': train_data,
            'test': test_data,
            'permutation': perm,
            'name': f"Permutation {t}"
        })
    return tasks


def make_split_cifar100_tasks(train_data, test_data, n_tasks=10):
    """Split-CIFAR-100: 10 classes per task"""
    tasks = []
    classes_per_task = 100 // n_tasks
    
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = [i for i, (_, y) in enumerate(train_data) if y in task_classes]
        test_idx = [i for i, (_, y) in enumerate(test_data) if y in task_classes]
        
        tasks.append({
            'train': Subset(train_data, train_idx),
            'test': Subset(test_data, test_idx),
            'classes': task_classes,
            'name': f"Classes {task_classes[0]}-{task_classes[-1]}"
        })
    return tasks


# =============================================================================
# METHOD 1: BASELINE (Naive Sequential Training)
# =============================================================================

class BaselineMethod:
    """No protection against forgetting - just train sequentially"""
    
    def __init__(self, model: nn.Module, lr: float = 0.001):
        self.model = model.to(DEVICE)
        self.lr = lr
        
    def train_task(self, task_data, epochs: int = 5, batch_size: int = 64,
                   permutation: Optional[np.ndarray] = None):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loader = DataLoader(task_data, batch_size=batch_size, shuffle=True, num_workers=0)
        
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if permutation is not None:
                    x = x.view(x.size(0), -1)[:, permutation].view(x.size(0), 1, 28, 28)
                
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()
    
    def evaluate(self, task_data, permutation: Optional[np.ndarray] = None) -> float:
        self.model.eval()
        correct, total = 0, 0
        loader = DataLoader(task_data, batch_size=256, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if permutation is not None:
                    x = x.view(x.size(0), -1)[:, permutation].view(x.size(0), 1, 28, 28)
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0


# =============================================================================
# METHOD 2: EWC (Elastic Weight Consolidation)
# =============================================================================

class EWCMethod:
    """
    EWC (Kirkpatrick et al., 2017)
    Penalizes changing weights that were important for previous tasks.
    """
    
    def __init__(self, model: nn.Module, lr: float = 0.001, ewc_lambda: float = 400):
        self.model = model.to(DEVICE)
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}
        self.n_tasks = 0
        
    def _compute_fisher(self, task_data, n_samples: int = 200,
                        permutation: Optional[np.ndarray] = None):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        loader = DataLoader(task_data, batch_size=1, shuffle=True, num_workers=0)
        
        samples = 0
        for x, y in loader:
            if samples >= n_samples:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            if permutation is not None:
                x = x.view(x.size(0), -1)[:, permutation].view(x.size(0), 1, 28, 28)
            
            self.model.zero_grad()
            F.cross_entropy(self.model(x), y).backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            samples += 1
        
        for n in fisher:
            fisher[n] /= samples
        return fisher
    
    def _ewc_penalty(self):
        if self.n_tasks == 0:
            return 0.0
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.ewc_lambda * loss
    
    def train_task(self, task_data, epochs: int = 5, batch_size: int = 64,
                   permutation: Optional[np.ndarray] = None):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loader = DataLoader(task_data, batch_size=batch_size, shuffle=True, num_workers=0)
        
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if permutation is not None:
                    x = x.view(x.size(0), -1)[:, permutation].view(x.size(0), 1, 28, 28)
                
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y) + self._ewc_penalty()
                loss.backward()
                optimizer.step()
        
        # Update Fisher and optimal params
        new_fisher = self._compute_fisher(task_data, permutation=permutation)
        if self.n_tasks == 0:
            self.fisher = new_fisher
        else:
            for n in new_fisher:
                self.fisher[n] = self.fisher.get(n, 0) + new_fisher[n]
        
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        self.n_tasks += 1
    
    def evaluate(self, task_data, permutation: Optional[np.ndarray] = None) -> float:
        self.model.eval()
        correct, total = 0, 0
        loader = DataLoader(task_data, batch_size=256, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if permutation is not None:
                    x = x.view(x.size(0), -1)[:, permutation].view(x.size(0), 1, 28, 28)
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0


# =============================================================================
# METHOD 3: SGM (Stochastic Gradient Mutation with Locking)
# =============================================================================

class SGMMethod:
    """
    SGM: Evolutionary optimization with binary parameter locking.
    
    Key differences from gradient methods:
    - Uses mutation + selection instead of backprop for weight updates
    - Binary locks on converged parameters (permanent)
    - Gradient used only for importance estimation (what to lock)
    """
    
    def __init__(self, model: nn.Module, pop_size: int = 20, elite_k: int = 5,
                 mutation_rate: float = 0.02):
        self.model = model.to(DEVICE)
        self.pop_size = pop_size
        self.elite_k = elite_k
        self.mutation_rate = mutation_rate
        
        # Flatten parameters
        self.param_shapes = {}
        self.param_names = []
        total = 0
        for n, p in self.model.named_parameters():
            self.param_shapes[n] = p.shape
            self.param_names.append(n)
            total += p.numel()
        
        self.dim = total
        self.x = self._flatten()
        self.lock = torch.zeros(self.dim, dtype=torch.bool, device=DEVICE)
        self.n_tasks = 0
    
    def _flatten(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for n, p in self.model.named_parameters()])
    
    def _unflatten(self, flat: torch.Tensor):
        idx = 0
        for n, p in self.model.named_parameters():
            size = p.numel()
            p.data.copy_(flat[idx:idx+size].view(p.shape))
            idx += size
    
    def _fitness(self, x: torch.Tensor, loader, permutation, max_samples: int = 300) -> float:
        """Negative accuracy (we minimize)"""
        self._unflatten(x)
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for bx, by in loader:
                if total >= max_samples:
                    break
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                if permutation is not None:
                    bx = bx.view(bx.size(0), -1)[:, permutation].view(bx.size(0), 1, 28, 28)
                pred = self.model(bx).argmax(dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)
        
        return -correct / total
    
    def _mutate(self, x: torch.Tensor) -> torch.Tensor:
        child = x.clone()
        free = ~self.lock
        n_free = free.sum().item()
        if n_free == 0:
            return child
        
        n_mut = max(1, int(n_free * 0.2))
        free_idx = torch.where(free)[0]
        mut_idx = free_idx[torch.randperm(len(free_idx), device=DEVICE)[:n_mut]]
        child[mut_idx] += torch.randn(len(mut_idx), device=DEVICE) * self.mutation_rate
        return child
    
    def _importance(self, task_data, permutation, n_samples: int = 100) -> torch.Tensor:
        """Gradient-based importance for locking decisions"""
        self._unflatten(self.x)
        self.model.train()
        
        grad_acc = torch.zeros(self.dim, device=DEVICE)
        loader = DataLoader(task_data, batch_size=32, shuffle=True, num_workers=0)
        samples = 0
        
        for bx, by in loader:
            if samples >= n_samples:
                break
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            if permutation is not None:
                bx = bx.view(bx.size(0), -1)[:, permutation].view(bx.size(0), 1, 28, 28)
            
            self.model.zero_grad()
            F.cross_entropy(self.model(bx), by).backward()
            
            idx = 0
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    grad_acc[idx:idx+p.numel()] += p.grad.view(-1).abs()
                idx += p.numel()
            samples += bx.size(0)
        
        return grad_acc / max(1, samples // 32)
    
    def train_task(self, task_data, generations: int = 50,
                   permutation: Optional[np.ndarray] = None):
        loader = DataLoader(task_data, batch_size=64, shuffle=True, num_workers=0)
        
        # Initialize population
        pop = [self.x.clone() for _ in range(self.pop_size)]
        best_fit, best_x = float('inf'), self.x.clone()
        
        for gen in range(generations):
            # Evaluate
            fits = torch.tensor([self._fitness(p, loader, permutation) for p in pop])
            
            # Track best
            if fits.min() < best_fit:
                best_fit = fits.min().item()
                best_x = pop[fits.argmin()].clone()
            
            # Elite selection + mutation
            elite_idx = fits.argsort()[:self.elite_k]
            elites = [pop[i].clone() for i in elite_idx]
            
            pop = elites.copy()
            while len(pop) < self.pop_size:
                parent = elites[torch.randint(len(elites), (1,)).item()]
                pop.append(self._mutate(parent))
        
        # Update solution
        self.x = best_x.clone()
        self._unflatten(self.x)
        
        # Lock important params
        importance = self._importance(task_data, permutation)
        threshold = importance.mean() + importance.std()
        new_locks = (importance > threshold) & (~self.lock)
        self.lock = self.lock | new_locks
        
        self.n_tasks += 1
        return self.lock.sum().item()
    
    def evaluate(self, task_data, permutation: Optional[np.ndarray] = None) -> float:
        self._unflatten(self.x)
        self.model.eval()
        correct, total = 0, 0
        loader = DataLoader(task_data, batch_size=256, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if permutation is not None:
                    x = x.view(x.size(0), -1)[:, permutation].view(x.size(0), 1, 28, 28)
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0
    
    def saturation(self) -> float:
        return 100 * self.lock.sum().item() / self.dim


# =============================================================================
# BENCHMARK RUNNERS
# =============================================================================

def run_benchmark(benchmark_name: str, task_fn, model_fn, methods: dict,
                  n_seeds: int = 3, epochs: int = 3, sgm_gens: int = 30,
                  verbose: bool = True):
    """Generic benchmark runner"""
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"{'='*70}")
    
    results = {name: {'final_acc': [], 'backward_transfer': [], 'matrices': []}
               for name in methods}
    
    for name, (method_class, kwargs) in methods.items():
        print(f"\n--- {name} ---")
        
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            tasks = task_fn()
            n_tasks = len(tasks)
            model = model_fn()
            method = method_class(model, **kwargs)
            
            acc_matrix = np.zeros((n_tasks, n_tasks))
            
            for t, task in enumerate(tasks):
                perm = task.get('permutation', None)
                if perm is not None:
                    perm = torch.from_numpy(perm).long().to(DEVICE)
                
                # Train
                if isinstance(method, SGMMethod):
                    locked = method.train_task(task['train'], generations=sgm_gens,
                                               permutation=perm)
                    if verbose and seed == 0:
                        print(f"  Task {t}: {method.saturation():.1f}% locked", end="")
                else:
                    method.train_task(task['train'], epochs=epochs, permutation=perm)
                    if verbose and seed == 0:
                        print(f"  Task {t}", end="")
                
                # Evaluate all tasks
                for j in range(t + 1):
                    perm_j = tasks[j].get('permutation', None)
                    if perm_j is not None:
                        perm_j = torch.from_numpy(perm_j).long().to(DEVICE)
                    acc_matrix[t][j] = method.evaluate(tasks[j]['test'], permutation=perm_j)
                
                if verbose and seed == 0:
                    print(f" -> {acc_matrix[t][t]*100:.1f}%")
            
            # Metrics
            final_acc = acc_matrix[-1].mean()
            bwt = sum(acc_matrix[-1][j] - acc_matrix[j][j] for j in range(n_tasks-1)) / (n_tasks-1)
            
            results[name]['final_acc'].append(final_acc)
            results[name]['backward_transfer'].append(bwt)
            results[name]['matrices'].append(acc_matrix)
            
            if verbose:
                print(f"  Seed {seed}: Final={final_acc*100:.1f}%, BWT={bwt*100:+.1f}%")
    
    # Summary
    print(f"\n{'-'*50}")
    print(f"{benchmark_name} SUMMARY")
    print(f"{'-'*50}")
    print(f"{'Method':<12} | {'Final Acc':<20} | {'Backward Transfer':<20}")
    print(f"{'-'*55}")
    
    for name in methods:
        acc = results[name]['final_acc']
        bwt = results[name]['backward_transfer']
        print(f"{name:<12} | {np.mean(acc)*100:5.1f}% +/- {np.std(acc)*100:4.1f}% | "
              f"{np.mean(bwt)*100:+5.1f}% +/- {np.std(bwt)*100:4.1f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    all_results = {}
    
    # Load datasets once
    print("\nLoading datasets...")
    mnist_train, mnist_test = get_mnist()
    print("  MNIST loaded")
    cifar_train, cifar_test = get_cifar100()
    print("  CIFAR-100 loaded")
    
    # Method configurations
    mnist_methods = {
        'Baseline': (BaselineMethod, {'lr': 0.001}),
        'EWC': (EWCMethod, {'lr': 0.001, 'ewc_lambda': 400}),
        'SGM': (SGMMethod, {'pop_size': 20, 'elite_k': 5, 'mutation_rate': 0.02})
    }
    
    cifar_methods = {
        'Baseline': (BaselineMethod, {'lr': 0.001}),
        'EWC': (EWCMethod, {'lr': 0.001, 'ewc_lambda': 1000}),
        'SGM': (SGMMethod, {'pop_size': 15, 'elite_k': 3, 'mutation_rate': 0.01})
    }
    
    # =========================================================================
    # BENCHMARK 1: Split-MNIST
    # =========================================================================
    all_results['split_mnist'] = run_benchmark(
        "SPLIT-MNIST (5 tasks, 2 classes each)",
        task_fn=lambda: make_split_mnist_tasks(mnist_train, mnist_test, 5),
        model_fn=lambda: MLP(784, 256, 10),
        methods=mnist_methods,
        n_seeds=N_SEEDS,
        epochs=3,
        sgm_gens=30
    )
    
    # =========================================================================
    # BENCHMARK 2: Permuted-MNIST
    # =========================================================================
    all_results['permuted_mnist'] = run_benchmark(
        "PERMUTED-MNIST (10 permutations)",
        task_fn=lambda: make_permuted_mnist_tasks(mnist_train, mnist_test, 10),
        model_fn=lambda: MLP(784, 256, 10),
        methods=mnist_methods,
        n_seeds=N_SEEDS,
        epochs=2,
        sgm_gens=25
    )
    
    # =========================================================================
    # BENCHMARK 3: Split-CIFAR-100
    # =========================================================================
    all_results['split_cifar100'] = run_benchmark(
        "SPLIT-CIFAR-100 (10 tasks, 10 classes each)",
        task_fn=lambda: make_split_cifar100_tasks(cifar_train, cifar_test, 10),
        model_fn=lambda: CNN(100),
        methods=cifar_methods,
        n_seeds=1,  # Fewer seeds for CIFAR (slower)
        epochs=5,
        sgm_gens=15
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for bench_name, results in all_results.items():
        print(f"\n{bench_name.upper()}:")
        for method_name in results:
            acc = results[method_name]['final_acc']
            bwt = results[method_name]['backward_transfer']
            print(f"  {method_name:<10}: {np.mean(acc)*100:5.1f}% +/- {np.std(acc)*100:.1f}%  "
                  f"(BWT: {np.mean(bwt)*100:+.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    
    # Statistical significance
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (SGM vs EWC)")
    print("="*70)
    
    for bench_name, results in all_results.items():
        if 'SGM' in results and 'EWC' in results:
            sgm_acc = np.array(results['SGM']['final_acc'])
            ewc_acc = np.array(results['EWC']['final_acc'])
            
            if len(sgm_acc) > 1:
                diff = sgm_acc - ewc_acc
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)
                
                # Simple t-test approximation
                t_stat = mean_diff / (std_diff / np.sqrt(len(diff))) if std_diff > 0 else 0
                
                print(f"\n{bench_name}:")
                print(f"  SGM - EWC = {mean_diff*100:+.2f}% +/- {std_diff*100:.2f}%")
                print(f"  t-statistic: {t_stat:.2f}")
                if abs(t_stat) > 2.0:
                    winner = "SGM" if mean_diff > 0 else "EWC"
                    print(f"  -> {winner} significantly better (p < 0.05)")
                else:
                    print(f"  -> No significant difference")
    
    return all_results


if __name__ == "__main__":
    results = main()