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

Gradient interference is exactly 0.000000 by orthogonal construction.

| Test | Retention | Interference |
|------|-----------|-------------|
| 20 sequential tasks | 1.0000x | 0.000000 |
| 100 sequential tasks | 1.0000x | 0.000000 |
| 1000 sequential tasks | 1.0000x | 0.000000 |
| Baseline (no locking) | 0.0002x | 5285.65x worse |

## Structure

```
core/sgm_core_primitives.py, sgm_demo.py, sgm_model_primitives.py
experiments/academic_validation.py, mnist_cifar_combined.py, real_benchmarks.py, split_mnist.py
tests/comprehensive.py, extreme_scale.py, dense_overlap.py, ultimate_stress.py
```

## Quick Start

```bash
python core/sgm_demo.py
python tests/extreme_scale.py
```

## Author

**Andrew C. Dorman**

## License

Proprietary License. See [LICENSE](LICENSE).