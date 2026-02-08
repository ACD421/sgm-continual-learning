# SGM: Sparse Geometric Mutation for Interference-Free Continual Learning

**Author:** Andrew Dorman ([Hollow Point Labs](https://github.com/ACD421))

## Abstract

Sparse Geometric Mutation (SGM) introduces **convergence-based binary locking** as a substrate-level primitive for continual learning. When a parameter converges during training on a task, its coordinate is permanently locked -- removed from the computation graph for all future tasks. This yields **mathematically zero gradient interference** between tasks, not approximately zero, but exactly zero: locked coordinates cannot be updated because they no longer participate in backpropagation.

The mechanism is three lines of code. No regularization penalties, no replay buffers, no architectural expansion, no task identifiers at inference time.

## Key Results

All numbers are real, measured, and reproducible. Run the corresponding test to verify.

### Binary Locking (Core Primitive)

| Property | Value |
|----------|-------|
| Retention after 20 tasks | **1.0000x** (perfect) |
| Retention after 100 tasks | **1.0000x** (perfect) |
| Retention after 1000 tasks | **1.0000x** (perfect) |
| Forgetting ratio vs. baseline (20 tasks) | **5285.65x worse without locking** |
| Gradient interference | **0.000000** |
| Plasticity (alpha) | **0.035 -- 0.060** |

### Benchmark Comparisons

The `experiments/real_benchmarks.py` suite compares SGM against standard continual learning baselines (EWC, naive fine-tuning) on Split-MNIST, Permuted-MNIST, and Split-CIFAR-100 using PyTorch.

### Additional Validated Properties

| Property | Value |
|----------|-------|
| NAND gate truth table accuracy | 4/4 |
| Gate locking consistency (std across reloads) | 0.000 |
| Post-quantization retention | Preserved (see `tests/quantization.py`) |

## Architecture

SGM operates at the parameter level. The core abstraction:

1. **Train** a task on a shared parameter vector
2. **Detect convergence** per coordinate (magnitude exceeds threshold)
3. **Lock** converged coordinates (binary mask, permanently frozen)
4. **Repeat** for the next task -- locked coordinates are excluded from gradients

This is not a regularization method. Locked parameters are physically removed from the optimization, guaranteeing zero interference by construction.

### Why It Works

Traditional continual learning methods (EWC, SI, PackNet) add soft penalties or prune after training. SGM locks *during* training at the moment of convergence. The lock is binary (not weighted), permanent (not decayed), and applied at the individual parameter level (not layer or block level, though block-level variants are also provided).

## Quick Start

```bash
# Clone the repo
git clone https://github.com/ACD421/sgm-continual-learning.git
cd sgm-continual-learning

# Run the self-contained demo (no dependencies beyond NumPy)
python core/sgm_demo.py

# Run the core primitive tests
python core/sgm_core_primitives.py

# Run real PyTorch benchmarks (requires torch, torchvision)
python experiments/real_benchmarks.py
```

## File Guide

### `core/` -- Foundation

| File | Description |
|------|-------------|
| `sgm_core_primitives.py` | SparseRegionTask, SGMBaseline, SGMWithLocking. Non-overlapping, partial overlap, random mask, and contradictory task tests with full statistical evaluation. |
| `sgm_model_primitives.py` | NNModel (2-layer FF), TransformerModel (elementwise self-attention), SGMBaselineModel, SGMWithLockingModel. Flattened param vectors with synthetic regression. |
| `sgm_demo.py` | Clean self-contained demo. Best entry point for understanding the system. |

### `experiments/` -- Research Experiments

| File | Description |
|------|-------------|
| `split_mnist.py` | Synthetic MNIST with hand-crafted digit prototypes, noise/shift augmentation, block-level locking. |
| `real_benchmarks.py` | PyTorch benchmark suite: Split-MNIST, Permuted-MNIST, Split-CIFAR-100. Compares Baseline vs EWC vs SGM. |
| `masked_forward_isolation.py` | MaskedMLP with forward-isolation via weight masks. Proves zero cross-task interference in the forward pass. |
| `academic_validation.py` | Paper-ready validation: synthetic + real benchmarks, baseline comparisons, ablation studies, confidence intervals. |
| `academic_validation_v2.py` | Fixed version: coalition detection removed, dynamic budget to prevent saturation collapse, per-layer diagnostics. |
| `plasticity_amplification.py` | Tests whether remaining free dimensions show larger update magnitudes as more dimensions lock. |
| `mnist_cifar_combined.py` | MNIST MLP (1024x1024) + CIFAR CNN with block-level locking and visualization. |

### `tests/` -- Stress Tests and Validation

| File | Description |
|------|-------------|
| `realworld_stress.py` | NLP embedding, vision feature, and RL policy task simulations with causal scoring. |
| `comprehensive.py` | Parameter scaling, extreme task count (20+), saturation, adversarial, overlap, noise perturbation. |
| `extreme_scale.py` | 500 tasks, 10M params, real embedding dims (768/1024/4096), capacity saturation to 99.9%. |
| `ultimate_stress.py` | 1000+ tasks, parameter scaling to 10^9, resource/latency profiling. |
| `quantization.py` | Post-training quantization resilience: n-bit uniform symmetric quantization on locked models. |
| `dense_overlap.py` | Fully overlapping digit tasks (all features active). Intentionally challenging for SGM. |
| `sklearn_digits.py` | SGM on sklearn handwritten digits (64 features, 10 classes, real data). |
| `overlap_adaptation.py` | Overlapping MNIST tasks with shared digit 0 across all tasks. |
| `hierarchical_task_sim.py` | CIFAR-like hierarchical + NLP bag-of-words synthetic tasks with overlapping input masks. |
| `extended_scenarios.py` | Inference-time sparsity, incremental personalization, hybrid NN+logistic models. |
| `model_scaling.py` | Inference sparsity and personalization across NN and Transformer model types. |
| `cifar_pytorch_template.py` | CIFAR-10 via PyTorch SimpleNet MLP. Template for torch-based SGM benchmarks. |
| `split_mnist_minimal.py` | Minimal Split-MNIST with PyTorch MLP (256 hidden, 2-class). Clean reference implementation. |

## How SGM Differs from Prior Work

| Method | Mechanism | Interference | Memory Overhead | Task ID at Inference |
|--------|-----------|-------------|-----------------|---------------------|
| EWC (Kirkpatrick 2017) | Fisher penalty on important params | Reduced, not zero | O(params) per task | No |
| PackNet (Mallya 2018) | Post-hoc pruning + freezing | Zero for frozen | Binary mask per task | Yes |
| Progressive Nets | New columns per task | Zero | O(params) per task | Yes |
| **SGM (this work)** | **Convergence-based binary lock** | **Zero by construction** | **Single binary mask** | **No** |

## Citation

If you use this work in your research, please cite:

```bibtex
@software{dorman2026sgm,
  author = {Dorman, Andrew},
  title = {SGM: Sparse Geometric Mutation for Interference-Free Continual Learning},
  year = {2026},
  url = {https://github.com/ACD421/sgm-continual-learning}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
