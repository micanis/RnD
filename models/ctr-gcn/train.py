# src/ctr-gcn/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from data.ntu_reader import read_skeleton
from data.preprocess import to_tensor, bones, motion
from data.edges import EDGES
from model.ctrgcn import CTRGCN
from config import NTU_V, INPUT_T, NUM_CLASS_60


# ===============================
# Dataset
# ===============================
class NTUDataset(Dataset):
    def __init__(self, root: Path, modality="joint"):
        self.root = root
        self.files = sorted(list(root.glob("*.skeleton")))
        self.modality = modality

    def __len__(self): return len(self.files)

    def _label(self, name):
        a = int(name[name.find("A")+1:name.find("A")+4])
        return a - 1  # 1～60 → 0～59

    def __getitem__(self, idx):
        fp = self.files[idx]
        frames = read_skeleton(fp)
        x = to_tensor(frames)   # (C,T,V,M)
        x = x[:,:,:,0]          # M=1 に固定

        if self.modality == "joint":
            x_mod = x
        elif self.modality == "bone":
            x_mod = bones(x, EDGES)
        elif self.modality == "joint-motion":
            x_mod = motion(x)
        elif self.modality == "bone-motion":
            x_mod = motion(bones(x, EDGES))
        else:
            raise ValueError

        y = self._label(fp.name)
        return torch.from_numpy(x_mod), y


# ===============================
# Train / Eval
# ===============================
def train_loop(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0; acc = 0; loss_sum = 0

    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        out = model(x)
        loss = ce(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        total += x.size(0)
        acc += (pred == y).sum().item()
        loss_sum += loss.item() * x.size(0)

    return loss_sum / total, acc / total


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0; acc = 0; loss_sum = 0

    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        out = model(x)
        loss = ce(out, y)

        pred = out.argmax(1)
        total += x.size(0)
        acc += (pred == y).sum().item()
        loss_sum += loss.item() * x.size(0)

    return loss_sum / total, acc / total


# ===============================
# Main
# ===============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_root = Path("src/ctr-gcn/datasets/ntu/nturgb+d_skeletons")
    dataset = NTUDataset(train_root, modality="joint")

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    model = CTRGCN(V=NTU_V, num_class=NUM_CLASS_60).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1, 6):  # ★まずは動作確認のため5 epoch
        loss, acc = train_loop(model, loader, optimizer, device)
        print(f"[Epoch {epoch}] loss={loss:.4f}, acc={acc:.4f}")

    torch.save(model.state_dict(), "ctr-gcn.pth")
    print("Saved model → ctr-gcn.pth")


if __name__ == "__main__":
    main()
