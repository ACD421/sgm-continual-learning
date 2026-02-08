import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# ===========================
# Settings
# ===========================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 3
BLOCK_SIZE = 64
SGM_LOCK_FRACTION = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TASKS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===========================
# Dataset helpers
# ===========================
def get_mnist_task(task_id):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    targets = np.array(dataset.targets)
    digits = [task_id * 2, task_id * 2 + 1]
    idx = np.isin(targets, digits)
    data = dataset.data[idx].float().view(-1, 28*28)/255.0
    targets = targets[idx]
    targets = np.where(targets == digits[0], 0, 1)
    return data, targets

def get_cifar_task(task_id):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    targets = np.array(dataset.targets)
    digits = [task_id*2, task_id*2+1]
    idx = np.isin(targets, digits)
    data = dataset.data[idx].astype(np.float32)/255.0
    data = torch.tensor(data).permute(0,3,1,2)
    targets = targets[idx]
    targets = np.where(targets == digits[0], 0, 1)
    return data, targets

# ===========================
# Models
# ===========================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024,1024], output_size=2):
        super().__init__()
        layers=[]
        last_size=input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size,h))
            layers.append(nn.ReLU())
            last_size=h
        layers.append(nn.Linear(last_size,output_size))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self,input_channels=3,output_size=2):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*8*8,256)
        self.fc2=nn.Linear(256,output_size)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

# ===========================
# SGM helpers
# ===========================
def get_flat_params_and_masks(model):
    params=[]
    masks=[]
    for p in model.parameters():
        params.append(p.view(-1))
        masks.append(torch.ones_like(p.view(-1)))
    flat_params=torch.cat(params)
    flat_mask=torch.cat(masks)
    return flat_params, flat_mask

def update_sgm_mask(model, current_mask, lock_fraction=SGM_LOCK_FRACTION):
    # flatten all gradients
    grads=[p.grad.view(-1).abs() for p in model.parameters() if p.grad is not None]
    flat_grads=torch.cat(grads)
    num_params=len(flat_grads)
    num_lock=int(lock_fraction*num_params)
    if num_lock==0:
        return current_mask
    top_indices=torch.topk(flat_grads,num_lock).indices
    new_mask=current_mask.clone()
    new_mask[top_indices]=0.0
    return new_mask

# ===========================
# Training & Evaluation
# ===========================
def train_model(model,data,targets,sgm_mask=None):
    model.train()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    dataset=torch.utils.data.TensorDataset(data,torch.tensor(targets))
    loader=torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    for _ in range(EPOCHS):
        for x_batch,y_batch in loader:
            x_batch,y_batch=x_batch.to(DEVICE),y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits=model(x_batch)
            loss=F.cross_entropy(logits,y_batch)
            loss.backward()
            if sgm_mask is not None:
                offset=0
                for p in model.parameters():
                    numel=p.numel()
                    if p.grad is not None:
                        p.grad.view(-1).mul_(sgm_mask[offset:offset+numel].to(DEVICE))
                    offset+=numel
            optimizer.step()

def evaluate_model(model,data,targets):
    model.eval()
    with torch.no_grad():
        x,y=data.to(DEVICE),torch.tensor(targets).to(DEVICE)
        logits=model(x)
        preds=logits.argmax(dim=1)
        acc=(preds==y).float().mean().item()
    return acc

# ===========================
# Master Test
# ===========================
def master_test(dataset_name,input_size,get_task_fn,num_tasks=NUM_TASKS,model_type="mlp"):
    print(f"\n=== {dataset_name} MASTER TEST ===")
    if dataset_name=="MNIST":
        baseline_model=MLP(input_size).to(DEVICE)
        sgm_model=MLP(input_size).to(DEVICE)
    else:
        baseline_model=SmallCNN().to(DEVICE)
        sgm_model=SmallCNN().to(DEVICE)

    baseline_retention=[]
    sgm_retention=[]
    locked_perc=[]
    # initialize SGM mask
    _,sgm_mask=get_flat_params_and_masks(sgm_model)

    for t in range(num_tasks):
        data,targets=get_task_fn(t)
        # baseline
        train_model(baseline_model,data,targets)
        acc_base=evaluate_model(baseline_model,data,targets)
        baseline_retention.append(round(acc_base,3))
        # SGM
        train_model(sgm_model,data,targets,sgm_mask)
        acc_sgm=evaluate_model(sgm_model,data,targets)
        sgm_retention.append(round(acc_sgm,3))
        # update mask
        sgm_mask=update_sgm_mask(sgm_model,sgm_mask)
        locked_perc.append(round(100*(1.0-sgm_mask.mean().item()),1))
        print(f"Task {t} baseline done. Acc: {acc_base:.3f}")
        print(f"Task {t} SGM done. Acc: {acc_sgm:.3f}, Locked: {locked_perc[-1]}%")

    # plots
    try:
        plt.figure()
        plt.plot(range(num_tasks),baseline_retention,label="Baseline")
        plt.plot(range(num_tasks),sgm_retention,label="SGM")
        plt.xlabel("Task")
        plt.ylabel("Retention")
        plt.title(f"{dataset_name} Retention vs Task")
        plt.legend()
        plt.savefig(f"{dataset_name}_retention_vs_task.png")
        plt.close()

        plt.figure()
        plt.plot(range(num_tasks),locked_perc,label="SGM Locked %")
        plt.xlabel("Task")
        plt.ylabel("Locked %")
        plt.title(f"{dataset_name} Locked Parameters Over Time")
        plt.legend()
        plt.savefig(f"{dataset_name}_locked_percent.png")
        plt.close()
    except:
        print("Plotting failed, summary will still be shown.")

    # final summary
    print(f"\n--- FINAL SUMMARY for {dataset_name} ---")
    print(f"BASELINE Retention: {baseline_retention}")
    print(f"SGM Retention: {sgm_retention}")
    print(f"Locked % over tasks: {locked_perc}")
    print("="*40)

# ===========================
# Run
# ===========================
if __name__=="__main__":
    master_test("MNIST",28*28,get_mnist_task)
    master_test("CIFAR",32*32*3,get_cifar_task)
    print("\n=== MASTER TEST COMPLETE ===")
