<div align="center">

# SGM: Sparse Geometric Mutation

### Interference-Free Continual Learning Primitive

**1.0000x retention | Zero gradient interference | Three lines of code**

[![Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

</div>

## The Solution

```python
if lock_mask[i] == True:
    delta[i] = 0  # This dimension CANNOT change.
```

Gradient interference is exactly 0.000000 by orthogonal construction. A binary mask permanently freezes learned structure. Locked dimensions are immutable -- not regularized, not penalized, not approximated. Frozen. The result is mathematically guaranteed zero forgetting on all locked parameters, at any scale, for any number of sequential tasks.

## Results

All numbers are real and reproducible. See [VALIDATED_RESULTS.md](VALIDATED_RESULTS.md) for full tables.

| Test | Retention | Baseline | Script |
|------|-----------|----------|--------|
| 20 sequential tasks | 1.0000x | varies | `tests/comprehensive.py` |
| 100 sequential tasks | 1.0000x | 0.0002x | `tests/extreme_scale.py` |
| 1000 sequential tasks | 1.0000x | 12.79x degradation | `tests/ultimate_stress.py` |
| 10M parameters | 1.0000x | 1.35x degradation | `tests/model_scaling.py` |
| 12-layer transformer sim | 1.0000x every layer | up to 5.04x | `tests/extreme_scale.py` |
| Gradient interference | 0.000000 | 5285.65x worse | `core/sgm_demo.py` |
| Academic validation | 9/9 PASS | -- | `experiments/academic_validation.py` |
| MNIST (MLP, 5 tasks) | 0.992-1.000 acc | 0.998-1.000 acc | `experiments/mnist_cifar_combined.py` |
| CIFAR-10 (SmallCNN, 5 tasks) | 0.827-0.932 acc | 0.823-0.930 acc | `experiments/mnist_cifar_combined.py` |

## Quick Start

```bash
# Install dependencies
pip install numpy scipy torch torchvision scikit-learn

# Run the demo (no GPU required)
python core/sgm_demo.py

# Run the full stress test suite
python tests/extreme_scale.py

# Run real-data experiments (requires CUDA GPU)
python experiments/mnist_cifar_combined.py
```

Expected output from `core/sgm_demo.py`:
```
Task 1: trained, locked 160/5000 dims
Task 2: trained, locked 320/5000 dims
...
Task 20: trained, locked 3200/5000 dims
Baseline retention: 0.0002x (catastrophic forgetting)
Locked retention:   1.0000x (perfect)
Gradient interference: 0.000000
```

## Structure

```
core/
    sgm_core_primitives.py    Core SGM locking primitive and coalition detection
    sgm_model_primitives.py   Model-level primitives (MLP, CNN integration)
    sgm_demo.py               One-command proof of concept

experiments/
    academic_validation.py    9-test formal validation suite (orthogonality, masking, retention)
    academic_validation_v2.py Updated academic validation with additional edge cases
    mnist_cifar_combined.py   Real MNIST + CIFAR-10 on PyTorch/CUDA
    split_mnist.py            Sequential split-MNIST with per-task locking
    masked_forward_isolation.py  Hard-lock checksum byte-level verification
    plasticity_amplification.py  Per-dimension plasticity analysis under progressive locking
    real_benchmarks.py        Real-world stress scenarios (distribution shift, noise, adversarial)

tests/
    comprehensive.py          Full test suite: non-overlapping, overlapping, random, contradictory
    extreme_scale.py          500 tasks, 10M params, 12-layer transformer sim, adversarial overlap
    ultimate_stress.py        1000 tasks, oscillating targets, capacity saturation, edge cases
    dense_overlap.py          Dense digit tasks with high-dimensional overlap
    extended_scenarios.py     Extended evaluation with additional task configurations
    model_scaling.py          Parameter scaling tests up to 10M dimensions
    overlap_adaptation.py     Adaptation and evolution under overlapping task constraints
    quantization.py           Post-training 8-bit quantization impact on locked retention
    realworld_stress.py       Real-world stress simulation with distribution shift
    hierarchical_task_sim.py  Hierarchical image and language task simulation
    cifar_pytorch_template.py CIFAR-10 PyTorch template with SGM integration
    sklearn_digits.py         Real sklearn digits dataset with SGM locking
    split_mnist_minimal.py    Minimal split-MNIST sequential learning test
```

## Experiments

| Script | Description |
|--------|-------------|
| `academic_validation.py` | 9-test formal validation suite covering orthogonality, masking invariants, and retention proofs |
| `academic_validation_v2.py` | Updated validation with additional edge cases and stricter assertions |
| `mnist_cifar_combined.py` | Real MNIST + CIFAR-10 on PyTorch/CUDA with per-task locking percentages |
| `split_mnist.py` | Sequential split-MNIST with progressive locking after each digit pair |
| `extreme_scale.py` | 500 tasks, 10M params, 12-layer transformer simulation, adversarial overlap |
| `masked_forward_isolation.py` | Hard-lock checksum verification: byte-level invariant on locked parameters |
| `plasticity_amplification.py` | Per-dimension plasticity analysis showing update amplitude on free dimensions |
| `real_benchmarks.py` | Real-world stress scenarios: distribution shift, noise injection, adversarial perturbation |

## Tests

| Script | Description |
|--------|-------------|
| `comprehensive.py` | Non-overlapping, overlapping, random mask, and contradictory task tests |
| `extreme_scale.py` | 500 tasks (5000 dims), 10M parameters, transformer sim, adversarial |
| `ultimate_stress.py` | 1000 tasks, oscillating targets, capacity saturation, failure mode probing |
| `dense_overlap.py` | Dense digit tasks with high-dimensional feature overlap |
| `extended_scenarios.py` | Extended evaluation with varied task configurations |
| `model_scaling.py` | Parameter count scaling from 1K to 10M dimensions |
| `overlap_adaptation.py` | Adaptation and evolution under overlapping constraints |
| `quantization.py` | 8-bit quantization: zero additional degradation on locked retention |
| `realworld_stress.py` | Real-world stress simulation with distribution shift and noise |
| `hierarchical_task_sim.py` | Hierarchical image and language task simulation |
| `cifar_pytorch_template.py` | CIFAR-10 PyTorch integration template |
| `sklearn_digits.py` | Real sklearn digits dataset evaluation |
| `split_mnist_minimal.py` | Minimal split-MNIST sequential learning test |

## Author

**Andrew C. Dorman**

## License

Proprietary License. See [LICENSE](LICENSE).
