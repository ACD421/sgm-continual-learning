<div align="center">

# SGM Continual Learning

**Binary locking: 1.0000x retention, zero gradient interference, three lines of code.**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%204070-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

## The Primitive

```python
if lock_mask[i] == True:
    delta[i] = 0  # This dimension CANNOT change.
```

When a parameter converges, it locks. Gradient becomes zero. That coordinate is removed
from the computation graph entirely. The next task trains on remaining free dimensions.
Forgetting cannot happen because locked coordinates cannot move. Not approximately. Exactly.

## Results

| Test | Tasks | Locked | Baseline | Script |
|------|-------|--------|----------|--------|
| Sequential retention | 1000 | 1.0000x | 12.79x degradation | `tests/ultimate_stress.py` |
| Gradient interference | -- | 0.000000 | EWC: 6.14 | `tests/test_interference.py` |
| Academic validation | 9 tests | 9/9 PASS | -- | `experiments/academic_validation.py` |
| Hard-lock checksum | 5 tasks | Byte-exact | -- | `experiments/masked_forward_isolation.py` |
| Real MNIST (PyTorch) | 5 | 0.992-1.000 | 0.996-1.000 | `experiments/mnist_cifar_combined.py` |
| Real CIFAR-10 (PyTorch) | 5 | 0.827-0.932 | 0.823-0.930 | `experiments/mnist_cifar_combined.py` |
| Parameter scaling | 10M | 1.00x | 1.35x | `tests/extreme_scale.py` |
| 12-layer transformer | 12 layers | 1.00x all | 1.00-5.04x | `tests/extreme_scale.py` |
| 8-bit quantization | -- | No degradation | -- | `tests/quantization.py` |
| Adversarial oscillation | 100 osc. | 1.0000x | -- | `tests/ultimate_stress.py` |

See [VALIDATED_RESULTS.md](VALIDATED_RESULTS.md) for complete tables.

## Experiments

| Script | What It Tests |
|--------|--------------|
| `experiments/academic_validation.py` | 9 formal tests: invariance, isolation, scale, orthogonality |
| `experiments/mnist_cifar_combined.py` | Real MNIST + CIFAR-10 on GPU with PyTorch |
| `experiments/split_mnist.py` | Sequential digit pairs with retention tracking |
| `experiments/masked_forward_isolation.py` | SHA-256 checksum proof of exact parameter preservation |
| `experiments/plasticity_amplification.py` | Per-dimension learning rate amplification on free dims |
| `experiments/real_benchmarks.py` | Extended real-data benchmarks |

## Tests (14 scripts)

| Script | What It Verifies |
|--------|-----------------|
| `tests/extreme_scale.py` | 500 tasks, 10M params, 12-layer transformer, adversarial |
| `tests/ultimate_stress.py` | 1000 tasks, oscillating targets, edge cases |
| `tests/realworld_stress.py` | Task overlap, cross-domain transfer, perturbation |
| `tests/dense_overlap.py` | Full feature overlap with sklearn digits |
| `tests/model_scaling.py` | NN + transformer scaling, inference sparsity |
| `tests/quantization.py` | 8-bit quantization survival |
| `tests/hierarchical_task_sim.py` | CIFAR-like and NLP-like hierarchical tasks |
| `tests/sklearn_digits.py` | Real sklearn digits, 64-32-16-10 NN |

## Quick Start

```bash
git clone https://github.com/ACD421/sgm-continual-learning.git
cd sgm-continual-learning
pip install -r requirements.txt

# Core proof
python -c "
from core.sgm_core_primitives import SGMWithLocking, SparseRegionTask
import numpy as np
np.random.seed(42)
model = SGMWithLocking(dim=100)
task1 = SparseRegionTask(100, (0.0, 0.5), seed=1)
task2 = SparseRegionTask(100, (0.5, 1.0), seed=2)
model.step(task1, 1000)
t1_loss = task1.loss(model.best_x)
model.step(task2, 1000)
print(f'Locked dims: {int(np.sum(model.lock > 0.5))}/100')
print(f'Task 1 retention: {task1.loss(model.best_x)/t1_loss:.4f}x')
"

# Full test suite
python tests/extreme_scale.py --full
python experiments/academic_validation.py
python experiments/mnist_cifar_combined.py  # Needs GPU
```

## Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0          # For MNIST/CIFAR experiments
torchvision>=0.15.0   # For real data loading
scikit-learn>=1.3.0   # For sklearn digits tests
```

## Author

**Andrew Dorman** -- Independent AI researcher, Southlake TX
- GitHub: [ACD421](https://github.com/ACD421)

## License

Proprietary. See [LICENSE](LICENSE) for terms.
