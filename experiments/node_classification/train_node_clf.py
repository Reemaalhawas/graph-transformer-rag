# experiments/node_classification/train_node_clf.py
"""
Node type classification on the STaRK-Prime knowledge graph.
Tests whether GNN architecture advantage on QA generalises to node classification.

Two modes:
  text      — uses full 1536-d ada-002 embeddings (expected ~90%+ accuracy)
  structure — uses 7-d one-hot node type encoding (forces structure-only reasoning)

Usage: python experiments/node_classification/train_node_clf.py
"""
import os, sys, json, copy, torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.gnn_variants import GCN, GraphTransformer

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA    = "/home/ubuntu/stark-graphrag-storage/node_clf_data.pt"
CKPTS   = "/home/ubuntu/stark-graphrag-storage/node_clf_checkpoints"
RESULTS = "/home/ubuntu/stark-graphrag-storage/node_clf_results"
N_CLS   = 7


class NodeClassifier(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        self.clf = nn.Linear(gnn.out_channels, N_CLS)

    def forward(self, x, edge_index, edge_attr=None):
        return self.clf(self.gnn(x, edge_index, edge_attr, batch=None))


def build_gnn(arch, in_ch):
    if arch == "gat":
        g = GAT(in_channels=in_ch, hidden_channels=512,
                num_layers=4, out_channels=512)
        g.out_channels = 512
        return g
    if arch == "gcn":
        return GCN(in_channels=in_ch, hidden_channels=512,
                   num_layers=4, out_channels=512)
    if arch == "graph_transformer":
        return GraphTransformer(in_channels=in_ch, hidden_channels=512,
                                num_layers=4, heads=4, out_channels=512,
                                edge_dim=None)
    raise ValueError(f"Unknown arch: {arch}")


def train_one(model, loader, opt, device):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index)
        mask = batch.y[:batch.batch_size] >= 0
        loss = F.cross_entropy(
            out[:batch.batch_size][mask],
            batch.y[:batch.batch_size][mask]
        )
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out[:batch.batch_size].argmax(-1)
        mask = batch.y[:batch.batch_size] >= 0
        preds.extend(pred[mask].cpu().tolist())
        labels.extend(batch.y[:batch.batch_size][mask].cpu().tolist())
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return round(acc, 4), round(f1, 4)


def run(arch, data, mode, epochs=30, patience=5):
    print(f"\n{'='*55}")
    print(f"  arch={arch}  mode={mode}")
    print(f"{'='*55}")

    d = copy.copy(data)
    if mode == "structure":
        d.x = F.one_hot(data.y.clamp(min=0), num_classes=N_CLS).float()
        in_ch = N_CLS
    else:
        in_ch = data.x.shape[1]  # 1536

    model = NodeClassifier(build_gnn(arch, in_ch)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def mkl(mask):
        return NeighborLoader(d, num_neighbors=[10, 5], batch_size=64,
                              input_nodes=mask, shuffle=(mask is d.train_mask))

    train_l = mkl(d.train_mask)
    val_l   = mkl(d.val_mask)
    test_l  = mkl(d.test_mask)

    best, ctr = 0, 0
    os.makedirs(CKPTS, exist_ok=True)
    ckpt = f"{CKPTS}/{arch}_{mode}_best.pt"

    for ep in range(1, epochs + 1):
        loss = train_one(model, train_l, opt, DEVICE)
        va, vf = evaluate(model, val_l, DEVICE)
        print(f"Epoch {ep:3d} | loss={loss:.4f} | val_acc={va:.4f} | val_f1={vf:.4f}")
        if va > best:
            best, ctr = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            ctr += 1
            if ctr >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=False))
    ta, tf = evaluate(model, test_l, DEVICE)
    print(f"TEST acc={ta:.4f} | f1={tf:.4f}")
    return {"arch": arch, "mode": mode, "test_acc": ta, "test_f1": tf, "best_val_acc": best}


if __name__ == "__main__":
    data = torch.load(DATA, weights_only=False)
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    all_r = []
    for arch in ["gat", "gcn", "graph_transformer"]:
        for mode in ["text", "structure"]:
            all_r.append(run(arch, data, mode))

    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/results.json", "w") as f:
        json.dump(all_r, f, indent=2)

    print("\n=== FINAL NODE CLASSIFICATION RESULTS ===")
    for r in all_r:
        print(f"  {r['arch']:20s} {r['mode']:10s} "
              f"acc={r['test_acc']} f1={r['test_f1']}")
