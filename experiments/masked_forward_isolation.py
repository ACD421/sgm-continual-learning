import os, time, math, random, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ----------------------------
# Determinism (best effort)
# ----------------------------
def seed_all(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ----------------------------
# Data: Split MNIST tasks (pairs)
# ----------------------------
def get_split_mnist_tasks(
    root: str = "./data",
    batch_size: int = 256,
    task_pairs=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
    seed: int = 1234,
):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    def subset_for_digits(ds, digits):
        idx = [i for i, (_, y) in enumerate(ds) if int(y) in digits]
        return Subset(ds, idx)

    tasks = []
    for (a, b) in task_pairs:
        tr = subset_for_digits(train, (a, b))
        te = subset_for_digits(test, (a, b))
        g = torch.Generator()
        g.manual_seed(seed)
        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
        te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)
        tasks.append(((a, b), tr_loader, te_loader))
    return tasks


# ----------------------------
# Masked model: forward isolation
# ----------------------------
class MaskedMLP(nn.Module):
    """
    Forward uses masked weights:
        W_eff = W * mask_W
        b_eff = b * mask_b

    This is the critical fix: parameters outside the task mask contribute exactly zero
    to the forward pass, so changing them cannot affect a task evaluated under its mask.
    """
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.fc1_w = nn.Parameter(torch.empty(hidden, 28*28))
        self.fc1_b = nn.Parameter(torch.empty(hidden))
        self.fc2_w = nn.Parameter(torch.empty(hidden, hidden))
        self.fc2_b = nn.Parameter(torch.empty(hidden))
        self.fc3_w = nn.Parameter(torch.empty(10, hidden))
        self.fc3_b = nn.Parameter(torch.empty(10))

        self.reset_parameters()

        # masks (buffers) default to all-ones = normal network
        self.register_buffer("m_fc1_w", torch.ones_like(self.fc1_w))
        self.register_buffer("m_fc1_b", torch.ones_like(self.fc1_b))
        self.register_buffer("m_fc2_w", torch.ones_like(self.fc2_w))
        self.register_buffer("m_fc2_b", torch.ones_like(self.fc2_b))
        self.register_buffer("m_fc3_w", torch.ones_like(self.fc3_w))
        self.register_buffer("m_fc3_b", torch.ones_like(self.fc3_b))

        # soft-lock scaling mask (optional): multiplies gradients via forward scaling
        # if you want soft updates to "locked" coords, you can set masks to tiny values instead of 0/1.
        # For hard isolation keep masks binary.
        self._soft_scale = 1.0

    def reset_parameters(self):
        # Kaiming init similar to Linear layers
        nn.init.kaiming_uniform_(self.fc1_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc3_w, a=math.sqrt(5))
        for b in (self.fc1_b, self.fc2_b, self.fc3_b):
            fan_in = 1
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    @torch.no_grad()
    def set_task_masks(self, masks: Dict[str, torch.Tensor]):
        # masks are tensors shaped like params, with values in {0,1} for hard isolation
        self.m_fc1_w.copy_(masks["fc1_w"])
        self.m_fc1_b.copy_(masks["fc1_b"])
        self.m_fc2_w.copy_(masks["fc2_w"])
        self.m_fc2_b.copy_(masks["fc2_b"])
        self.m_fc3_w.copy_(masks["fc3_w"])
        self.m_fc3_b.copy_(masks["fc3_b"])

    def forward(self, x):
        x = x.view(x.size(0), -1)

        w1 = self.fc1_w * self.m_fc1_w
        b1 = self.fc1_b * self.m_fc1_b
        h1 = F.relu(F.linear(x, w1, b1))

        w2 = self.fc2_w * self.m_fc2_w
        b2 = self.fc2_b * self.m_fc2_b
        h2 = F.relu(F.linear(h1, w2, b2))

        w3 = self.fc3_w * self.m_fc3_w
        b3 = self.fc3_b * self.m_fc3_b
        out = F.linear(h2, w3, b3)
        return out


# ----------------------------
# Substrate: block masks over ALL params
# ----------------------------
@dataclass
class ParamView:
    name: str
    tensor: torch.Tensor
    numel: int
    start: int
    end: int


class MaskSubstrate:
    """
    Flattens *all* parameters into a 1D substrate and can create task masks
    selecting disjoint blocks.

    Key idea:
      - Task mask selects coordinates that are ACTIVE in forward (mask=1),
        all other coordinates are forced to 0 in forward.
      - Therefore tasks are forward-isolated by construction.
    """
    def __init__(self, model: MaskedMLP, block_size: int = 2048, device: str = "cpu"):
        self.model = model
        self.device = device
        self.block_size = int(block_size)

        # build flat mapping
        params = [
            ("fc1_w", model.fc1_w),
            ("fc1_b", model.fc1_b),
            ("fc2_w", model.fc2_w),
            ("fc2_b", model.fc2_b),
            ("fc3_w", model.fc3_w),
            ("fc3_b", model.fc3_b),
        ]

        self.views: List[ParamView] = []
        offset = 0
        for name, p in params:
            n = p.numel()
            self.views.append(ParamView(name=name, tensor=p, numel=n, start=offset, end=offset+n))
            offset += n

        self.total_params = offset
        self.num_blocks = math.ceil(self.total_params / self.block_size)

        # Track hard locks (coords that are committed permanently)
        self.locked = torch.zeros(self.total_params, dtype=torch.bool, device=device)

    def blocks_for_task(self, t: int, blocks_per_task: int) -> List[int]:
        start = t * blocks_per_task
        end = min((t+1)*blocks_per_task, self.num_blocks)
        return list(range(start, end))

    def make_task_mask(self, active_blocks: List[int], soft_locked_scale: Optional[float]=None) -> Dict[str, torch.Tensor]:
        """
        Returns per-parameter masks for MaskedMLP.set_task_masks().
        - If soft_locked_scale is None: binary masks (hard isolation).
        - If soft_locked_scale is provided: locked coords get scale in (0..1), active coords are 1, inactive are 0.
          (This enables "soft lock drift" experiments.)
        """
        m_flat = torch.zeros(self.total_params, dtype=torch.float32, device=self.device)
        # active coords = 1
        for bid in active_blocks:
            a = bid * self.block_size
            b = min((bid+1)*self.block_size, self.total_params)
            m_flat[a:b] = 1.0

        # optional: allow tiny contribution/update on locked coords
        if soft_locked_scale is not None:
            s = float(soft_locked_scale)
            if not (0.0 <= s <= 1.0):
                raise ValueError("soft_locked_scale must be in [0,1]")
            # coords that are locked but not active for this task contribute at scale s (instead of 0)
            locked_not_active = self.locked & (m_flat == 0)
            m_flat[locked_not_active] = s

        # reshape into per-parameter masks
        masks = {}
        for v in self.views:
            sl = m_flat[v.start:v.end].view_as(v.tensor)
            masks[v.name] = sl
        return masks

    def hard_lock_blocks(self, blocks: List[int]):
        for bid in blocks:
            a = bid * self.block_size
            b = min((bid+1)*self.block_size, self.total_params)
            self.locked[a:b] = True

    def locked_fraction(self) -> float:
        return float(self.locked.float().mean().item())

    def active_coords_for_mask(self, masks: Dict[str, torch.Tensor]) -> int:
        # counts coords with mask==1 (active)
        total = 0
        for k, m in masks.items():
            total += int((m == 1.0).sum().item())
        return total

    @staticmethod
    def logits_checksum(logits: torch.Tensor) -> str:
        arr = logits.detach().cpu().contiguous().numpy().astype(np.float32)
        return hashlib.sha256(arr.tobytes()).hexdigest()


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def eval_task(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, torch.Tensor]:
    model.eval()
    correct = 0
    total = 0
    logits_all = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        logits_all.append(logits.detach().cpu())
    return correct / max(1, total), torch.cat(logits_all, dim=0)


def train_one_task(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    steps = 0
    t0 = time.perf_counter()
    for _ in range(int(epochs)):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            steps += 1
    t1 = time.perf_counter()
    return {"steps": steps, "seconds": float(t1 - t0)}


# ----------------------------
# TEST 1: HARD invariant (forward-isolated)
# ----------------------------
def test_hard_lock_invariant(
    device="cpu",
    seed=1234,
    hidden=512,
    block_size=2048,
    blocks_per_task=16,
    epochs_per_task=1,
    lr=3e-4,
):
    print("\n=== TEST 1: HARD-LOCK INVARIANT (correct forward isolation) ===")
    seed_all(seed)

    model = MaskedMLP(hidden=hidden).to(device)
    sub = MaskSubstrate(model, block_size=block_size, device=device)
    tasks = get_split_mnist_tasks(seed=seed)

    # Train anchor task on its mask
    anchor_digits, anchor_train, anchor_test = tasks[0]
    b0 = sub.blocks_for_task(0, blocks_per_task)
    m0 = sub.make_task_mask(b0)           # binary mask
    model.set_task_masks(m0)

    train_one_task(model, anchor_train, device, epochs_per_task, lr)
    sub.hard_lock_blocks(b0)              # commit anchor coords

    acc0, logits0 = eval_task(model, anchor_test, device)
    chk0 = sub.logits_checksum(logits0)
    print(f"Anchor {anchor_digits} acc: {acc0*100:.2f}% | checksum: {chk0[:16]}... | locked={sub.locked_fraction()*100:.3f}%")

    # Train later tasks on disjoint masks; anchor evaluated under m0 must not change
    for t in range(1, len(tasks)):
        digits, tr, _ = tasks[t]
        bt = sub.blocks_for_task(t, blocks_per_task)
        mt = sub.make_task_mask(bt)
        model.set_task_masks(mt)

        train_one_task(model, tr, device, epochs_per_task, lr)
        sub.hard_lock_blocks(bt)

        # re-eval anchor under anchor mask (THIS is the invariant check)
        model.set_task_masks(m0)
        acc_now, logits_now = eval_task(model, anchor_test, device)
        chk_now = sub.logits_checksum(logits_now)

        same = (chk_now == chk0)
        print(f"After task {t} {digits}: anchor acc {acc_now*100:.2f}% | checksum match: {same}")
        if not same:
            print("FAIL: checksum changed. This should not happen under forward-isolated hard masks.")
            return False

    print("PASS: Exact retention verified (forward-isolated hard mask).")
    return True


# ----------------------------
# TEST 2: Soft drift (allow tiny locked contribution)
# ----------------------------
def test_soft_lock_drift(
    device="cpu",
    seed=1234,
    hidden=512,
    block_size=2048,
    blocks_per_task=16,
    epochs_per_task=1,
    lr=3e-4,
    soft_locked_scale=1e-3,
):
    print("\n=== TEST 2: SOFT-LOCK DRIFT (controlled plasticity) ===")
    seed_all(seed)

    model = MaskedMLP(hidden=hidden).to(device)
    sub = MaskSubstrate(model, block_size=block_size, device=device)
    tasks = get_split_mnist_tasks(seed=seed)

    anchor_digits, anchor_train, anchor_test = tasks[0]
    b0 = sub.blocks_for_task(0, blocks_per_task)
    m0 = sub.make_task_mask(b0)
    model.set_task_masks(m0)

    train_one_task(model, anchor_train, device, epochs_per_task, lr)
    sub.hard_lock_blocks(b0)

    acc0, logits0 = eval_task(model, anchor_test, device)
    chk0 = sub.logits_checksum(logits0)
    print(f"Anchor {anchor_digits} acc: {acc0*100:.2f}% | checksum: {chk0[:16]}...")

    # Soft mode: each new task mask is active on its blocks, but previously locked coords leak in at tiny scale
    for t in range(1, len(tasks)):
        digits, tr, _ = tasks[t]
        bt = sub.blocks_for_task(t, blocks_per_task)
        mt = sub.make_task_mask(bt, soft_locked_scale=soft_locked_scale)
        model.set_task_masks(mt)

        train_one_task(model, tr, device, epochs_per_task, lr)
        sub.hard_lock_blocks(bt)

        # Evaluate anchor under a soft mask that leaks locked coords (so drift can appear)
        m0_soft = sub.make_task_mask(b0, soft_locked_scale=soft_locked_scale)
        model.set_task_masks(m0_soft)
        acc_now, logits_now = eval_task(model, anchor_test, device)
        chk_now = sub.logits_checksum(logits_now)
        print(f"After task {t} {digits}: anchor acc {acc_now*100:.2f}% | checksum changed: {chk_now != chk0}")

    print("DONE: Soft mode shows drift by design.")
    return True


# ----------------------------
# TEST 3: Compute signal (active coords count + time)
# ----------------------------
def test_compute_signal(
    device="cpu",
    seed=1234,
    hidden=512,
    block_size=2048,
    blocks_per_task=16,
    epochs_per_task=1,
    lr=3e-4,
):
    print("\n=== TEST 3: COMPUTE SIGNAL (active coords + time) ===")
    seed_all(seed)

    model = MaskedMLP(hidden=hidden).to(device)
    sub = MaskSubstrate(model, block_size=block_size, device=device)
    tasks = get_split_mnist_tasks(seed=seed)

    for t, (digits, tr, te) in enumerate(tasks):
        bt = sub.blocks_for_task(t, blocks_per_task)
        mt = sub.make_task_mask(bt)
        model.set_task_masks(mt)

        info = train_one_task(model, tr, device, epochs_per_task, lr)
        sub.hard_lock_blocks(bt)

        acc, _ = eval_task(model, te, device)
        active = sub.active_coords_for_mask(mt)
        print(f"Task {t} {digits} | acc {acc*100:.2f}% | active_coords {active:,} | locked {sub.locked_fraction()*100:.3f}% | time {info['seconds']:.2f}s")

    print("NOTE: Wall-time may not drop unless you use sparse kernels; active_coords is the honest compute proxy.")
    return True


# ----------------------------
# CLI
# ----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="all", choices=["all","hard","soft","compute"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--blocks_per_task", type=int, default=16)
    p.add_argument("--epochs_per_task", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--soft_locked_scale", type=float, default=1e-3)
    args = p.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    if args.test in ("all","hard"):
        ok = test_hard_lock_invariant(
            device=args.device, seed=args.seed, hidden=args.hidden,
            block_size=args.block_size, blocks_per_task=args.blocks_per_task,
            epochs_per_task=args.epochs_per_task, lr=args.lr,
        )
        if not ok:
            raise SystemExit(1)

    if args.test in ("all","soft"):
        test_soft_lock_drift(
            device=args.device, seed=args.seed, hidden=args.hidden,
            block_size=args.block_size, blocks_per_task=args.blocks_per_task,
            epochs_per_task=args.epochs_per_task, lr=args.lr,
            soft_locked_scale=args.soft_locked_scale,
        )

    if args.test in ("all","compute"):
        test_compute_signal(
            device=args.device, seed=args.seed, hidden=args.hidden,
            block_size=args.block_size, blocks_per_task=args.blocks_per_task,
            epochs_per_task=args.epochs_per_task, lr=args.lr,
        )

    print("\nAll requested tests finished.")


if __name__ == "__main__":
    main()
