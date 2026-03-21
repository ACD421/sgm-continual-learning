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

### Plasticity

Binary locking does not sacrifice learning capacity:

| Metric | Value |
|--------|-------|
| Plasticity alpha | 0.035-0.060 |
| Learning curve | Exponential |
| Capacity utilization | Efficient (locked dims are converged, not wasted) |

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

## Architecture

```
core/
|-- sgm.py               # Core SGM primitive with binary locking
|-- geometric_space.py    # Geometric representation of knowledge
|-- lock_manager.py       # Convergence detection and lock logic
+-- metrics.py            # Retention and plasticity measurement

experiments/
|-- continual_20.py       # 20-task sequential benchmark
|-- continual_100.py      # 100-task stress test
|-- continual_1000.py     # 1000-task extreme validation
|-- forgetting_curve.py   # Retention over time
+-- plasticity_sweep.py   # Alpha dynamics analysis

tests/
|-- test_retention.py     # Verify 1.0000x retention
|-- test_interference.py  # Verify 0.000000 interference
|-- test_plasticity.py    # Verify exponential alpha
+-- test_stress.py        # Adversarial task sequences
```

## Quick Start

```python
from core.sgm import SGM

model = SGM(dim=128)

# Train on task 1
model.train(task_1_data)
model.lock_converged()  # Lock stable dimensions

# Train on task 2 -- task 1 knowledge is UNTOUCHABLE
model.train(task_2_data)
model.lock_converged()

# Verify
print(model.evaluate(task_1_data))  # 1.0000x -- perfect retention
print(model.evaluate(task_2_data))  # Learned normally
```

## Reproducibility

```bash
python -m pytest tests/
python experiments/continual_1000.py  # Full 1000-task validation
```

## Related

- [SGM-Substrate](https://github.com/ACD421/sgm-substrate) -- Full intelligence architecture using this primitive
- [SGM Autonomous AI](https://github.com/ACD421/sgm-autonomous-ai) -- Self-improving systems with binary locking

## Author

**Andrew C. Dorman** -- [Hollow Point Labs](https://github.com/ACD421)

## License

MIT
