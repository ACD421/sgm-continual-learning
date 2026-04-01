# Validated Results

All numbers below are real, measured, and reproducible. Run the corresponding script to verify.

## Binary Locking Primitive

| Property | Value | Script |
|----------|-------|--------|
| Retention (20 tasks) | 1.0000x | tests/test_retention.py |
| Retention (100 tasks) | 1.0000x | tests/test_retention.py |
| Retention (1000 tasks) | 1.0000x | tests/ultimate_stress.py |
| Gradient interference | 0.000000 | tests/test_interference.py |
| Plasticity alpha | 0.035-0.062 | tests/test_plasticity.py |
| Hard-lock checksum invariant | PASS (exact byte match) | experiments/masked_forward_isolation.py |
| Academic validation | 9/9 PASS | experiments/academic_validation.py |

## Scale Tests

| Test | Baseline Degradation | Locked Retention |
|------|---------------------|-----------------|
| 500 tasks (5000 dims) | 2.35x | 1.00x |
| 1000 tasks (10000 dims) | 12.79x | 1.00x |
| 10M parameters | 1.35x | 1.00x |
| 12-layer transformer sim | up to 5.04x | 1.00x every layer |

## Real Data (PyTorch, CUDA)

| Dataset | Tasks | Baseline Acc | SGM Locked Acc | Locked % |
|---------|-------|-------------|----------------|----------|
| MNIST (MLP) | 5 | 0.998-1.000 | 0.992-1.000 | 10-47% |
| CIFAR-10 (SmallCNN) | 5 | 0.823-0.930 | 0.827-0.932 | 10-45% |

## Adversarial Tests

| Scenario | Result |
|----------|--------|
| 100% feature overlap | Locked wins (2.75 vs 17.55 at 100 tasks) |
| Oscillating targets (100 cycles) | 1.0000x retention |
| Contradictory tasks | Retention on locked half maintained |
| 2-sigma perturbation | Mask protects structure, not values |

## Quantization

8-bit quantization introduces zero additional degradation to locked retention.

## Capacity Saturation

| Locked % | Free Dims | Improvement over Baseline |
|----------|-----------|--------------------------|
| 90% | 1000 | 65.4% |
| 95% | 500 | 60.9% |
| 99% | 100 | 95.4% |
| 99.9% | 10 | 29.0% |
