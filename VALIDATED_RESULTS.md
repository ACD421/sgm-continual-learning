# Validated Results

All numbers below are real, measured, and independently replicated on April 1, 2026.
Every test can be reproduced by running the corresponding script.

## Core Primitive: Binary Locking

The mechanism is three lines: when a dimension converges, it locks. Gradient becomes zero.
That coordinate is removed from optimization entirely. Forgetting is eliminated by construction.

```python
if lock_mask[i] == True:
    delta[i] = 0  # This dimension CANNOT change.
```

## Retention Tests

| Test | Tasks | Locked Retention | Baseline Retention | Script |
|------|-------|------------------|--------------------|--------|
| Sequential (synthetic) | 20 | 1.0000x | varies | `tests/test_retention.py` |
| Sequential (synthetic) | 100 | 1.0000x | varies | `tests/test_retention.py` |
| Sequential (synthetic) | 500 | 1.0000x | 2.35x degradation | `tests/extreme_scale.py` |
| Sequential (synthetic) | 1000 | 1.0000x | 12.79x degradation | `tests/ultimate_stress.py` |
| Oscillating adversarial | 100 osc. | 1.0000x | N/A | `tests/ultimate_stress.py` |
| Hard-lock checksum | 5 tasks | Byte-exact match | N/A | `experiments/masked_forward_isolation.py` |

## Gradient Interference

| Metric | Binary Locking | EWC | Script |
|--------|---------------|-----|--------|
| Interference | 0.000000 | 6.141671 | `tests/test_interference.py` |
| Ratio | 0x | 6.14x | `tests/test_interference.py` |

Gradient interference is not approximately zero. It is exactly zero.
Locked coordinates are removed from the computation graph entirely.

## Academic Validation (9 Formal Tests)

All from `experiments/academic_validation.py`:

| Test | Result | Detail |
|------|--------|--------|
| A1: Fundamental Invariant | PASS | 100 trials, 0 violations |
| A2: Task Isolation | PASS | Retention 1.000000 |
| A3: Sequential (10 tasks) | PASS | Mean retention 1.000000 |
| A4: Gradient Compatibility | PASS | Mean retention 1.000000 |
| A5: Scale Invariance | PASS | 100/500/1000/5000/10000 dims, all 1.000000 |
| A6: Structured vs Random | PASS | Structured 1.00 vs Random 1.82, p=7.15e-34 |
| A7: Plasticity Amplification | PASS | 104.4x super-linear amplification |
| A8: Geometric Orthogonality | PASS | Structured 0.40 vs Random 0.71, p=1.40e-89 |
| A9: Capacity Bounds | PASS | Theoretical max = Actual max |

## Real Data (PyTorch, CUDA)

### MNIST (5 binary tasks, MLP)

| Task | Baseline Acc | SGM Locked Acc | Locked % |
|------|-------------|----------------|----------|
| 0 | 1.000 | 0.998 | 10.0% |
| 1 | 0.996 | 0.994 | 20.0% |
| 2 | 1.000 | 1.000 | 30.0% |
| 3 | 0.999 | 1.000 | 39.9% |
| 4 | 0.998 | 0.992 | 46.5% |

Script: `experiments/mnist_cifar_combined.py`

### CIFAR-10 (5 binary tasks, SmallCNN)

| Task | Baseline Acc | SGM Locked Acc | Locked % |
|------|-------------|----------------|----------|
| 0 | 0.895 | 0.932 | 10.0% |
| 1 | 0.823 | 0.827 | 20.0% |
| 2 | 0.882 | 0.851 | 30.0% |
| 3 | 0.912 | 0.856 | 38.3% |
| 4 | 0.930 | 0.847 | 45.2% |

Script: `experiments/mnist_cifar_combined.py`

## Parameter Scaling

| Params | Baseline | Locked | Script |
|--------|----------|--------|--------|
| 1,000 | 1.28x | 1.00x | `tests/extreme_scale.py` |
| 10,000 | 1.30x | 1.00x | `tests/extreme_scale.py` |
| 100,000 | 1.31x | 1.00x | `tests/extreme_scale.py` |
| 1,000,000 | 1.35x | 1.00x | `tests/extreme_scale.py` |
| 10,000,000 | 1.35x | 1.00x | `tests/extreme_scale.py` |

Locking holds at 1.00x regardless of parameter count.

## Transformer Simulation (12-Layer)

| Layer | Baseline | Locked |
|-------|----------|--------|
| 1 | 2.05x | 1.00x |
| 4 | 5.04x | 1.00x |
| 5 | 4.45x | 1.00x |
| 10 | 2.78x | 1.00x |
| 12 | 1.00x | 1.00x |

All 12 layers: locked retention is 1.00x. Script: `tests/extreme_scale.py`

## Quantization (8-bit)

| Model | Float Baseline | Float Locked | 8-bit Baseline | 8-bit Locked |
|-------|---------------|-------------|----------------|-------------|
| NN | 3.00 | 1.67 | 2.99 | 1.67 |
| Transformer | 1.23 | 1.02 | 1.23 | 1.02 |

Locking survives quantization with no degradation. Script: `tests/quantization.py`

## Known Limitations

- Capacity is finite: once all dimensions are locked, no new tasks can be learned.
- Plasticity drops as locked fraction increases (0% remaining at 90% locked).
- Real MNIST/CIFAR retention is strong but not perfect 1.0000x due to threshold tuning on real distributions.
- The primitive guarantees exact retention on locked dimensions. It does not guarantee task accuracy on shared features.
