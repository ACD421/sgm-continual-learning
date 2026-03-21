<div align="center">

# SGM: Sparse Geometric Mutation

### Interference-Free Continual Learning Primitive

**1.0000x retention | Zero gradient interference | Three lines of code**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

</div>

## The Problem

Catastrophic forgetting remains the central obstacle in continual learning. Every existing approach -- EWC, PackNet, replay buffers, progressive networks -- trades off between retention and plasticity. None actually eliminates interference.

## The Solution

**Convergence-based binary locking.** When parameters converge during training on a task, their coordinates are permanently locked -- removed from the computation graph entirely for all future tasks.

```python
if lock_mask[i] == True:
    delta[i] = 0  # This dimension CANNOT change.
```

This is not regularization. This is not approximation. The gradient interference is **exactly 0.000000** by orthogonal construction.

## Results

| Test | Retention | Interference |
|------|-----------|-------------|
| 20 sequential tasks | 1.0000x | 0.000000 |
| 100 sequential tasks | 1.0000x | 0.000000 |
| 1000 sequential tasks | 1.0000x | 0.000000 |
| Baseline (no locking) | 0.0002x | 5285.65x worse |

## How It Works

1. **Train** on task T_i normally
2. **Measure** per-dimension variance across training
3. **Lock** dimensions where variance < threshold (converged)
4. **Future tasks** cannot modify locked dimensions -- they are invisible to the optimizer

The key insight: converged dimensions have found their correct value. Allowing future gradients to modify them is pure noise. Locking them is not a constraint -- it is the recognition that learning is complete for those coordinates.

## Why This Is Different

| Approach | Mechanism | Interference | Exact? |
|----------|-----------|-------------|--------|
| EWC | Fisher regularization | Low | No (approximation) |
| PackNet | Binary masks | Zero (masked) | Yes, but wastes capacity |
| Replay | Experience buffer | Reduced | No |
| Progressive Networks | New columns | Zero | Yes, but O(n) memory |
| **Binary Locking** | **Lock converged dims** | **Zero** | **Yes, O(1) per lock** |

Binary locking achieves exact zero interference with O(1) cost per locked dimension and no capacity waste (locked dimensions are genuinely converged).

## Structure

```
core/
|-- sgm_core_primitives.py   # SGM primitive with coalition locking and ablation testing
|-- sgm_demo.py              # One-command proof: structure preservation breaks scaling curves
+-- sgm_model_primitives.py  # Neural network integration of binary locking

experiments/
|-- academic_validation.py   # Rigorous academic benchmark suite
|-- academic_validation_v2.py
|-- masked_forward_isolation.py  # Forward isolation verification
|-- mnist_cifar_combined.py  # Real dataset validation
|-- plasticity_amplification.py  # Plasticity dynamics analysis
|-- real_benchmarks.py       # Benchmark against EWC, PackNet, replay
+-- split_mnist.py           # Standard split-MNIST continual learning

tests/
|-- comprehensive.py         # Full test suite
|-- dense_overlap.py         # Overlapping task dimensions
|-- extreme_scale.py         # 1000-task stress test
|-- hierarchical_task_sim.py # Hierarchical task structures
|-- model_scaling.py         # Scaling behavior analysis
|-- overlap_adaptation.py    # Adaptive overlap handling
|-- quantization.py          # Post-training quantization effects
|-- realworld_stress.py      # Adversarial task sequences
|-- sklearn_digits.py        # Real data: sklearn digits
|-- split_mnist_minimal.py   # Minimal split-MNIST
+-- ultimate_stress.py       # Extreme adversarial conditions
```

## Quick Start

```python
from core.sgm_core_primitives import SGMWithLocking, SparseRegionTask
import numpy as np

model = SGMWithLocking(dim=128)

# Train on task 1
task1 = SparseRegionTask(128, (0.0, 0.3))
model.train_task(task1, n_evals=500)

# Train on task 2 -- task 1 knowledge is UNTOUCHABLE
task2 = SparseRegionTask(128, (0.3, 0.6))
model.train_task(task2, n_evals=500)

# Verify
print(f"Task 1 retention: {model.evaluate(task1):.4f}x")  # 1.0000x
```

Or run the one-command demo:

```bash
python core/sgm_demo.py
```

## Reproducibility

```bash
python tests/comprehensive.py       # Full validation suite
python tests/extreme_scale.py       # 1000-task stress test
python tests/dense_overlap.py       # Overlapping dimension tests
python experiments/real_benchmarks.py  # Comparison against baselines
```

## Related

- [SGM-Substrate](https://github.com/ACD421/sgm-substrate) -- Full intelligence architecture using this primitive
- [SGM Autonomous AI](https://github.com/ACD421/sgm-autonomous-ai) -- Self-improving systems with binary locking

## Author

**Andrew C. Dorman** -- [Hollow Point Labs](https://github.com/ACD421)

## License

MIT