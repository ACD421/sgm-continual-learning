import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
import time
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = [(0,1),(2,3),(4,5),(6,7),(8,9)]
EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# SIMPLE BASE MODEL
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# DATASET SPLITTER
# -----------------------------
def get_task_loader(digits):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    idx = [i for i,(x,y) in enumerate(dataset) if y in digits]
    subset = torch.utils.data.Subset(dataset, idx)

    return torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

def get_eval_loader(digits):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    idx = [i for i,(x,y) in enumerate(dataset) if y in digits]
    subset = torch.utils.data.Subset(dataset, idx)

    return torch.utils.data.DataLoader(subset, batch_size=512)

# -----------------------------
# TRAIN / EVAL
# -----------------------------
def train_task(model, loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        for x,y in loader:
            y = (y % 2).to(DEVICE)
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

def eval_task(model, loader):
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for x,y in loader:
            y = (y % 2).to(DEVICE)
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# -----------------------------
# MASTER TEST
# -----------------------------
def master_test():
    model = MLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    results = {
        "task_accuracy": defaultdict(dict),
        "retention": {},
        "forward_transfer": {},
        "timing": {},
        "parameters": sum(p.numel() for p in model.parameters())
    }

    start_time = time.time()

    for t, digits in enumerate(TASKS):
        train_loader = get_task_loader(digits)
        eval_loader = get_eval_loader(digits)

        t0 = time.time()
        train_task(model, train_loader, optimizer)
        results["timing"][f"task_{t}"] = time.time() - t0

        # Evaluate all tasks so far
        for past_t, past_digits in enumerate(TASKS[:t+1]):
            acc = eval_task(model, get_eval_loader(past_digits))
            results["task_accuracy"][f"task_{t}"][f"eval_{past_t}"] = acc

        # Retention on first task
        if t > 0:
            results["retention"][f"after_{t}"] = (
                results["task_accuracy"][f"task_{t}"]["eval_0"] /
                results["task_accuracy"]["task_0"]["eval_0"]
            )

    results["total_time"] = time.time() - start_time
    return results

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    out = master_test()
    with open("master_test_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("MASTER TEST COMPLETE")
    print(json.dumps(out, indent=2))
