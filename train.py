import argparse
import math
import os
import time
import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader
from stark_qa import load_qa
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric import seed_everything

from torch_geometric.nn import GAT, TransformerConv

from g_retriever import GRetriever  
from llm_wrapper import LLM  
from tqdm import tqdm

try:
    from compute_metrics import compute_metrics
except ImportError:
    def compute_metrics(x): print("Metrics calculation ready.")

from STaRKQADatasetGDS import STaRKQADataset
from STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset

# --- Graph Transformer Architecture ---
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=4, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList()
        # Initial layer
        self.convs.append(TransformerConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        # Output layer
        self.convs.append(TransformerConv(hidden_channels, out_channels, heads=1, concat=True, dropout=dropout))

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs[:-1]:
            x = torch.relu(conv(x, edge_index, edge_attr))
        return self.convs[-1](x, edge_index, edge_attr)

# --- Helper Functions ---
def get_loss(model, batch, model_save_name) -> Tensor:
    if model_save_name.startswith('llm'):
        return model(batch.question, batch.label, batch.desc)
    return model(batch.question, batch.x, batch.edge_index, batch.batch,
                 batch.label, batch.edge_attr, batch.desc)

def inference_step(model, batch, model_save_name):
    if model_save_name.startswith('llm'):
        return model.inference(batch.question, batch.desc)
    return model.inference(batch.question, batch.x, batch.edge_index,
                           batch.batch, batch.edge_attr, batch.desc)

def save_params_dict(model, save_path):
    state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(state_dict, save_path)

# --- Main Training ---
def train(args):
    seed_everything(42)
    load_dotenv('db.env', override=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path Management
    suffix = "frozen-llm" if args.freeze_llm else "llm"
    run_id = f"{args.gnn_type}_{suffix}_{args.llama_version}"
    root_path = f"stark_qa_v{args.retrieval_config_version}_{args.algo_config_version}"
    
    ckpt_dir = os.path.join(root_path, "checkpoints", run_id)
    subg_dir = os.path.join(root_path, "subgraphs", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(subg_dir, exist_ok=True)

    # Dataset Loading
    qa_dataset = load_qa("prime")
    train_set = STaRKQADataset(root_path, qa_dataset.get_subset('train'), args.retrieval_config_version, args.algo_config_version, split="train")
    test_set = STaRKQADataset(root_path, qa_dataset.get_subset('test'), args.retrieval_config_version, args.algo_config_version, split="test")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size)

    # Model Selection
    if args.gnn_type == 'gt':
        gnn = GraphTransformer(1536, args.gnn_hidden_channels, 1536, args.num_gnn_layers)
    else:
        gnn = GAT(in_channels=1536, hidden_channels=args.gnn_hidden_channels, out_channels=1536, num_layers=args.num_gnn_layers, heads=4)

    llm = LLM(model_name='meta-llama/Llama-3.1-8B-Instruct' if args.llama_version == 'llama3.1-8b' else args.llama_version)
    model = GRetriever(llm=llm, gnn=gnn).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Training
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = get_loss(model, batch, run_id)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_params_dict(model, os.path.join(ckpt_dir, "best_model.pt"))

    # Evaluation & Subgraph Saving
    model.eval()
    outputs = []
    for i, batch in enumerate(tqdm(test_loader, desc="Saving Subgraphs")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = inference_step(model, batch, run_id)
            outputs.append({'pred': pred, 'question': batch.question, 'label': batch.label})
            
            # Save actual subgraph tensor data
            torch.save({'x': batch.x.cpu(), 'edge_index': batch.edge_index.cpu()}, 
                       os.path.join(subg_dir, f"batch_{i}.pt"))

    torch.save(outputs, os.path.join(root_path, f"{run_id}_eval_outs.pt"))
    print(f"Done! Checkpoints in {ckpt_dir} and Subgraphs in {subg_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_type', type=str, choices=['gat', 'gt'], default='gt')
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--llama_version', type=str, default='llama3.1-8b')
    parser.add_argument('--retrieval_config_version', type=int, default=17)
    parser.add_argument('--algo_config_version', type=int, default=0)
    parser.add_argument('--freeze_llm', type=bool, default=False)
    train(parser.parse_args())