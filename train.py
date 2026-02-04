"""
G-Retriever Training Script - GAT Encoder
==========================================
Updated to work with pre-computed subgraphs from Google Drive on Lambda Labs.

Usage:
    python train.py --encoder gat --seed 42
    python train.py --encoder transformer --seed 42
"""

import os
import json
import gc
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    "gnn_in_channels": 1536,      # OpenAI embedding dimension
    "gnn_hidden": 128,
    "gnn_out": 256,
    "gnn_layers": 2,
    "gnn_heads": 4,
    "gnn_dropout": 0.1,
    
    # LLM
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    "load_in_4bit": True,
    
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    
    # Training
    "epochs": 3,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    
    # Data
    "max_length": 512,
    "max_new_tokens": 64,
}


# =============================================================================
# DATASET - Loads Pre-computed Subgraphs
# =============================================================================

class PrecomputedSubgraphDataset(Dataset):
    """
    Dataset that loads pre-computed subgraphs from .pt files.
    Each .pt file contains a PyG Data object with:
        - x: node features [num_nodes, 1536]
        - edge_index: [2, num_edges]
        - edge_attr: edge features [num_edges, 1536] (optional)
        - question: the query string
        - answer: list of answer node IDs or string
    """
    
    def __init__(
        self, 
        data_dir: str, 
        tokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Get all .pt files
        self.files = sorted([
            f for f in os.listdir(data_dir) 
            if f.endswith('.pt')
        ])
        
        print(f"[{split}] Found {len(self.files)} subgraphs in {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load pre-computed subgraph
        path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(path, weights_only=False)
        
        # Extract components
        x = data.x  # Node features
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
        question = data.question
        answer = data.answer
        
        # Convert answer to string
        if isinstance(answer, list):
            answer_str = " | ".join(str(a) for a in answer)
        else:
            answer_str = str(answer) if answer else ""
        
        # Create prompt
        prompt = f"Question: {question}\n\nAnswer:"
        
        # Tokenize
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        full_text = f"Question: {question}\n\nAnswer: {answer_str}"
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (mask prompt tokens with -100)
        labels = full_encoding['input_ids'].clone()
        prompt_len = prompt_encoding['attention_mask'].sum()
        labels[0, :prompt_len] = -100  # Don't compute loss on prompt
        
        return {
            'x': x.float(),
            'edge_index': edge_index.long(),
            'edge_attr': edge_attr.float() if edge_attr is not None else None,
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'question': question,
            'answer_str': answer_str,
        }


def collate_fn(batch):
    """Custom collate function to batch graphs and text together."""
    
    # Build list of PyG Data objects
    graphs = []
    for item in batch:
        g = Data(
            x=item['x'],
            edge_index=item['edge_index'],
        )
        if item['edge_attr'] is not None:
            g.edge_attr = item['edge_attr']
        graphs.append(g)
    
    # Batch graphs using PyG
    batched_graph = Batch.from_data_list(graphs)
    
    # Stack text tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'x': batched_graph.x,
        'edge_index': batched_graph.edge_index,
        'edge_attr': batched_graph.edge_attr if hasattr(batched_graph, 'edge_attr') else None,
        'batch': batched_graph.batch,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'questions': [item['question'] for item in batch],
        'answer_strs': [item['answer_str'] for item in batch],
    }


# =============================================================================
# GNN ENCODERS
# =============================================================================

class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder (baseline).
    Uses local neighborhood attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        ))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=True
            ))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                dropout=dropout,
                concat=False
            ))
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global mean pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder (novel contribution).
    Uses transformer-style global attention over graph.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        beta: bool = True  # Learnable skip connection
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(TransformerConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            beta=beta,
            concat=True
        ))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                hidden_channels * heads,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                beta=beta,
                concat=True
            ))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(TransformerConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                dropout=dropout,
                beta=beta,
                concat=False
            ))
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


def get_gnn_encoder(encoder_type: str, config: dict) -> nn.Module:
    """Factory function to get GNN encoder."""
    if encoder_type == "gat":
        return GATEncoder(
            in_channels=config["gnn_in_channels"],
            hidden_channels=config["gnn_hidden"],
            out_channels=config["gnn_out"],
            num_layers=config["gnn_layers"],
            heads=config["gnn_heads"],
            dropout=config["gnn_dropout"]
        )
    elif encoder_type == "transformer":
        return GraphTransformerEncoder(
            in_channels=config["gnn_in_channels"],
            hidden_channels=config["gnn_hidden"],
            out_channels=config["gnn_out"],
            num_layers=config["gnn_layers"],
            heads=config["gnn_heads"],
            dropout=config["gnn_dropout"]
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# =============================================================================
# G-RETRIEVER MODEL
# =============================================================================

class GRetriever(nn.Module):
    """
    G-Retriever: GNN encoder + LLM with LoRA.
    Encodes graph structure and injects it into LLM.
    """
    
    def __init__(
        self,
        encoder_type: str = "gat",
        config: dict = None
    ):
        super().__init__()
        
        config = config or DEFAULT_CONFIG
        self.config = config
        self.encoder_type = encoder_type
        
        # GNN encoder
        print(f"Creating {encoder_type.upper()} encoder...")
        self.gnn = get_gnn_encoder(encoder_type, config)
        
        # Quantization config
        bnb_config = None
        if config["load_in_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load LLM
        print(f"Loading LLM: {config['llm_name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config['llm_name'],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Apply LoRA
        print("Applying LoRA...")
        self.llm = prepare_model_for_kbit_training(self.llm)
        
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        
        # Projection: GNN output -> LLM embedding space
        llm_hidden = self.llm.config.hidden_size
        self.graph_proj = nn.Sequential(
            nn.Linear(config["gnn_out"], llm_hidden),
            nn.LayerNorm(llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )
        
        # Move projection to same device as LLM
        self.graph_proj = self.graph_proj.to(self.llm.device)
    
    def encode_graph(self, x, edge_index, batch):
        """Encode graph and project to LLM space."""
        # GNN encoding
        graph_emb = self.gnn(x, edge_index, batch)  # [batch_size, gnn_out]
        
        # Project to LLM space
        graph_emb = self.graph_proj(graph_emb)  # [batch_size, llm_hidden]
        
        return graph_emb.unsqueeze(1)  # [batch_size, 1, llm_hidden]
    
    def forward(self, x, edge_index, batch, input_ids, attention_mask, labels=None):
        # Move graph data to device
        device = self.llm.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Encode graph
        graph_emb = self.encode_graph(x, edge_index, batch)  # [B, 1, H]
        
        # Get text embeddings
        text_emb = self.llm.get_input_embeddings()(input_ids)  # [B, L, H]
        
        # Prepend graph embedding to text
        inputs_embeds = torch.cat([graph_emb, text_emb], dim=1)  # [B, 1+L, H]
        
        # Adjust attention mask
        graph_mask = torch.ones(
            (attention_mask.size(0), 1),
            device=device,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([graph_mask, attention_mask], dim=1)
        
        # Adjust labels (pad with -100 for graph token)
        if labels is not None:
            label_pad = torch.full(
                (labels.size(0), 1),
                -100,
                device=device,
                dtype=labels.dtype
            )
            labels = torch.cat([label_pad, labels], dim=1)
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(self, x, edge_index, batch, input_ids, attention_mask, max_new_tokens=64):
        device = self.llm.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Encode graph
        graph_emb = self.encode_graph(x, edge_index, batch)
        
        # Get text embeddings
        text_emb = self.llm.get_input_embeddings()(input_ids)
        
        # Combine
        inputs_embeds = torch.cat([graph_emb, text_emb], dim=1)
        
        # Attention mask
        graph_mask = torch.ones(
            (attention_mask.size(0), 1),
            device=device,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([graph_mask, attention_mask], dim=1)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        
        return outputs


# =============================================================================
# METRICS
# =============================================================================

def compute_exact_match(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute exact match accuracy."""
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.strip().lower()
        gt_clean = gt.strip().lower()
        if pred_clean == gt_clean or gt_clean in pred_clean:
            correct += 1
    return correct / len(predictions) * 100


def compute_hits_at_k(predictions: List[str], ground_truths: List[str], k: int = 1) -> float:
    """Compute Hits@K metric."""
    hits = 0
    for pred, gt in zip(predictions, ground_truths):
        # Split prediction into multiple answers
        pred_list = [p.strip().lower() for p in pred.split('|')][:k]
        gt_list = [g.strip().lower() for g in gt.split('|')]
        
        if any(g in ' '.join(pred_list) for g in gt_list):
            hits += 1
    
    return hits / len(predictions) * 100


def compute_f1(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute token-level F1 score."""
    f1_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = set(pred.lower().split())
        gt_tokens = set(gt.lower().split())
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores) * 100


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, epoch, config):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_data in pbar:
        optimizer.zero_grad()
        
        outputs = model(
            x=batch_data['x'],
            edge_index=batch_data['edge_index'],
            batch=batch_data['batch'],
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            labels=batch_data['labels'],
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config["max_grad_norm"]
        )
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    
    all_preds = []
    all_gts = []
    
    for batch_data in tqdm(dataloader, desc="Evaluating"):
        # Generate predictions
        outputs = model.generate(
            x=batch_data['x'],
            edge_index=batch_data['edge_index'],
            batch=batch_data['batch'],
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            max_new_tokens=config["max_new_tokens"],
        )
        
        # Decode
        preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the answer part
        clean_preds = []
        for pred in preds:
            if "Answer:" in pred:
                pred = pred.split("Answer:")[-1].strip()
            clean_preds.append(pred)
        
        all_preds.extend(clean_preds)
        all_gts.extend(batch_data['answer_strs'])
    
    # Compute metrics
    metrics = {
        'exact_match': compute_exact_match(all_preds, all_gts),
        'hits@1': compute_hits_at_k(all_preds, all_gts, k=1),
        'hits@5': compute_hits_at_k(all_preds, all_gts, k=5),
        'f1': compute_f1(all_preds, all_gts),
    }
    
    return metrics, all_preds, all_gts


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train G-Retriever with GAT or Transformer")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed/train',
                        help='Directory containing pre-computed subgraphs')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    # Model
    parser.add_argument('--encoder', type=str, default='gat', choices=['gat', 'transformer'],
                        help='GNN encoder type')
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='LLM model name')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['llm_name'] = args.llm
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.encoder}_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"G-Retriever Training")
    print(f"  Encoder: {args.encoder.upper()}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    print("\nCreating model...")
    model = GRetriever(encoder_type=args.encoder, config=config)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = PrecomputedSubgraphDataset(
        data_dir=args.data_dir,
        tokenizer=model.tokenizer,
        max_length=config['max_length']
    )
    
    # Split: 80% train, 10% val, 10% test
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Eval only mode
    if args.eval_only:
        print("\nRunning evaluation only...")
        test_metrics, _, _ = evaluate(model, test_loader, config)
        print(f"Test Metrics: {test_metrics}")
        return
    
    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(config['warmup_ratio'] * total_steps)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Training loop
    best_metric = 0
    history = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, config)
        print(f"Val Metrics: {val_metrics}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
        })
        
        # Save best model
        if val_metrics['hits@1'] > best_metric:
            best_metric = val_metrics['hits@1']
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'args': vars(args),
            }, os.path.join(output_dir, 'best_model.pt'))
            
            print(f"*** Saved best model with Hits@1: {best_metric:.2f} ***")
        
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    test_metrics, test_preds, test_gts = evaluate(model, test_loader, config)
    print(f"Test Metrics: {test_metrics}")
    
    # Save final results
    results = {
        'encoder': args.encoder,
        'seed': args.seed,
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'best_val_hits@1': best_metric,
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump({
            'predictions': test_preds[:100],  # Save first 100
            'ground_truths': test_gts[:100],
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()