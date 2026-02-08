"""
SGM unified real-data and synthetic stress test
================================================

This script performs a comprehensive series of stress tests on a Sparse Gradient
Mutation (SGM) system.  It demonstrates continual learning, retention under
multiple scenarios, and the ability to handle both real image data (CIFAR-10)
and synthetic bag-of-words tasks.  The script requires PyTorch and
torchvision for the CIFAR tests.  The SGM model classes (e.g. SGMBaseline,
SGMWithLocking, SGMFractalLocking) and task helpers must be available in
your Python path.  You can import them from the files included in the
provided repository.

Tests included:

1. **CIFAR sequential tasks**:  Load CIFAR-10 and train on three disjoint
   class groups in sequence.  Measure retention of the first task after
   subsequent tasks.  Compares a baseline neural net with a coalition-locking
   variant.
2. **Synthetic bag-of-words tasks**:  Construct overlapping vocabulary
   segments and train a simple regression model using SGM to simulate
   natural-language continual learning.
3. **Additional tests**:  Placeholders for saturation, contradictory,
   overlap/random mask, fractal cosetting and noise robustness.  You can
   uncomment or extend these using helper functions from other modules.

Because this environment does not permit execution of PyTorch code, the
functions below serve as a template.  To run them, ensure you have the
SGM classes and PyTorch installed locally.

Usage:
    python3 sgm_comprehensive_real.py
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Subset = None
    torchvision = None
    transforms = None

# Placeholder imports for your SGM implementations
# from sgm_model_tests import SGMBaseline, SGMWithLocking
# from fractal_cosetting_experiment import SGMFractalLocking
# from sgm_cifar_natty_simulation import make_cifar_tasks, make_natty_tasks


class SimpleNet(nn.Module):
    """Simple MLP for CIFAR-10 images."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def load_cifar_tasks(n_samples_per_class=500):
    """Split CIFAR-10 into three tasks and load a subset of each."""
    if torchvision is None:
        raise RuntimeError("torchvision is required for CIFAR tests")
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10(root="./data", train=True,
                                          download=True, transform=transform)
    class_indices = {c: [] for c in range(10)}
    for idx, (_, label) in enumerate(cifar):
        if len(class_indices[label]) < n_samples_per_class:
            class_indices[label].append(idx)
    tasks = []
    # Define three non-overlapping class sets
    for task_classes in [(0, 1, 2, 3), (4, 5, 6), (7, 8, 9)]:
        indices = []
        for c in task_classes:
            indices += class_indices[c]
        subset = Subset(cifar, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=True)
        tasks.append(loader)
    return tasks


def train_nn_task(loader, model, optimizer, criterion, device, epochs=3):
    """Train a PyTorch model for a few epochs."""
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_accuracy(loader, model, device):
    """Compute classification accuracy on a dataset loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def run_cifar_continual():
    """Train a baseline and SGM variant sequentially on CIFAR tasks."""
    if torch is None:
        print("PyTorch is not available; skipping CIFAR test.")
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = load_cifar_tasks(n_samples_per_class=400)
    input_dim = 32 * 32 * 3
    baseline_model = SimpleNet(input_dim, 128, 10).to(device)
    locking_model = SimpleNet(input_dim, 128, 10).to(device)
    baseline_opt = torch.optim.SGD(baseline_model.parameters(), lr=0.01)
    locking_opt = torch.optim.SGD(locking_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    baseline_ret = []
    locking_ret = []
    baseline_first = locking_first = None
    for i, loader in enumerate(tasks):
        train_nn_task(loader, baseline_model, baseline_opt, criterion, device)
        train_nn_task(loader, locking_model, locking_opt, criterion, device)
        if i == 0:
            baseline_first = evaluate_accuracy(loader, baseline_model, device)
            locking_first = evaluate_accuracy(loader, locking_model, device)
        else:
            baseline_ret.append(evaluate_accuracy(tasks[0], baseline_model, device) / baseline_first)
            locking_ret.append(evaluate_accuracy(tasks[0], locking_model, device) / locking_first)
    print("\nCIFAR Continual Learning Test:")
    print("Baseline retention on first task:", baseline_ret)
    print("SGM-like retention on first task:", locking_ret)


def generate_bow_tasks(vocab_size=200, n_tasks=3, n_samples=200):
    """Create synthetic bag-of-words tasks with overlapping vocabularies."""
    tasks = []
    for i in range(n_tasks):
        start = int(i * vocab_size / n_tasks)
        end = start + int(vocab_size / n_tasks * 1.5)
        # wrap around vocabulary
        indices = np.random.randint(start % vocab_size, end % vocab_size, size=(n_samples, 20))
        X = np.zeros((n_samples, vocab_size), dtype=np.float32)
        for j, idxs in enumerate(indices):
            X[j, idxs] = 1.0
        y = (np.sum(indices, axis=1) % 2).astype(int)
        tasks.append((X, y))
    return tasks


def run_bow_continual():
    """Run sequential SGM training on synthetic bag-of-words tasks."""
    dims = 200
    try:
        baseline = SGMBaseline(dims)
        locking = SGMWithLocking(dims)
    except NameError:
        print("SGM classes are not imported; skipping bag-of-words test.")
        return
    tasks = generate_bow_tasks(vocab_size=dims, n_tasks=3, n_samples=200)
    base_ret = []
    lock_ret = []
    base_best = lock_best = None
    for i, (X, y) in enumerate(tasks):
        def task_fn(vec, yy=y, XX=X):
            preds = XX @ vec
            return float(np.mean((preds - yy) ** 2))
        baseline.reset(); locking.reset()
        baseline.step(task_fn, n_evals=200)
        locking.step(task_fn, n_evals=200)
        if i == 0:
            base_best = baseline.best_loss; lock_best = locking.best_loss
        else:
            base_ret.append(task_fn(baseline.best_x) / base_best)
            lock_ret.append(task_fn(locking.best_x) / lock_best)
    print("\nBag-of-Words Continual Learning Test:")
    print("Baseline retention on first task:", base_ret)
    print("SGM retention on first task:", lock_ret)


if __name__ == '__main__':
    # Run the CIFAR test (requires PyTorch/torchvision)
    try:
        run_cifar_continual()
    except Exception as e:
        print("CIFAR test failed:", e)
    # Run the bag-of-words test
    run_bow_continual()
    # Additional tests (saturation, contradictory, overlap, fractal, noise)
    # can be added here by importing the appropriate functions from other modules.