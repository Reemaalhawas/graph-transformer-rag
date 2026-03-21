"""
G-Retriever Training Script
============================
Supports GAT, Transformer, and GCN encoders.
Supports baseline, subgraph_pruning, g_retriever, and pipeline modes.

Usage:
    python train.py --encoder gat --mode g_retriever --seed 42
    python train.py --encoder transformer --mode pipeline --seed 42
    python train.py --encoder gcn --mode subgraph_pruning --seed 42
    python train.py --mode baseline --seed 42
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
from torch_geometric.nn import GATConv, TransformerConv, GCNConv, global_mean_pool

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
        - node_texts: list of node text descriptions (optional)
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

        # Optional node text descriptions for pipeline mode
        node_texts = getattr(data, 'node_texts', None)

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
            'node_texts': node_texts,
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
        'node_texts': [item['node_texts'] for item in batch],
    }


# =============================================================================
# GNN ENCODERS
# =============================================================================

class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.
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
    Graph Transformer encoder.
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


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder.
    Uses spectral-based graph convolutions (Kipf & Welling 2017).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs  # Accept and ignore heads/beta for API compatibility
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
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
    elif encoder_type == "gcn":
        return GCNEncoder(
            in_channels=config["gnn_in_channels"],
            hidden_channels=config["gnn_hidden"],
            out_channels=config["gnn_out"],
            num_layers=config["gnn_layers"],
            dropout=config["gnn_dropout"]
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# =============================================================================
# SUBGRAPH PRUNING
# =============================================================================

def prune_subgraph(x, edge_index, batch, keep_ratio: float = 0.5):
    """
    Prune each graph to its top-`keep_ratio` nodes by degree.
    Returns new (x, edge_index, batch) with remapped node indices.
    """
    num_nodes = x.size(0)
    device = x.device

    # Compute undirected degree per node
    degrees = torch.zeros(num_nodes, device=device)
    if edge_index.size(1) > 0:
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))

    # Per-graph: keep top-50% nodes
    keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    for b in batch.unique():
        b_mask = (batch == b)
        b_indices = torch.where(b_mask)[0]
        b_degrees = degrees[b_indices]
        n_keep = max(1, int(len(b_indices) * keep_ratio))
        _, top_local = torch.topk(b_degrees, n_keep)
        keep_mask[b_indices[top_local]] = True

    # Build old-to-new node id mapping
    node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    node_map[keep_mask] = torch.arange(keep_mask.sum(), device=device)

    new_x = x[keep_mask]
    new_batch = batch[keep_mask]

    # Filter edges to kept nodes and remap
    if edge_index.size(1) > 0:
        src, dst = edge_index
        valid = keep_mask[src] & keep_mask[dst]
        new_edge_index = node_map[edge_index[:, valid]]
    else:
        new_edge_index = edge_index

    return new_x, new_edge_index, new_batch


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
        graph_emb = self.gnn(x, edge_index, batch)   # [batch_size, gnn_out]
        graph_emb = self.graph_proj(graph_emb)         # [batch_size, llm_hidden]
        return graph_emb.unsqueeze(1)                  # [batch_size, 1, llm_hidden]

    def forward(self, x, edge_index, batch, input_ids, attention_mask, labels=None):
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

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    @torch.no_grad()
    def generate(self, x, edge_index, batch, input_ids, attention_mask, max_new_tokens=64):
        """GNN + LLM generation (g_retriever / subgraph_pruning modes)."""
        device = self.llm.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        graph_emb = self.encode_graph(x, edge_index, batch)
        text_emb = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([graph_emb, text_emb], dim=1)

        graph_mask = torch.ones(
            (attention_mask.size(0), 1),
            device=device,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([graph_mask, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        return outputs

    @torch.no_grad()
    def generate_baseline(self, input_ids, attention_mask, max_new_tokens=64):
        """LLM-only generation — no graph encoding (baseline mode)."""
        device = self.llm.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        return outputs

    @torch.no_grad()
    def generate_pipeline(
        self,
        x, edge_index, batch,
        questions: List[str],
        node_texts: Optional[List] = None,
        max_new_tokens: int = 64,
    ):
        """
        GNN + LLM generation with node descriptions injected into the prompt
        (pipeline mode).
        """
        device = self.llm.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        # Encode graph
        graph_emb = self.encode_graph(x, edge_index, batch)  # [B, 1, H]

        # Build augmented prompts
        augmented_prompts = []
        unique_batches = batch.unique().tolist()
        for i, (b, q) in enumerate(zip(unique_batches, questions)):
            if node_texts and node_texts[i] is not None:
                texts = node_texts[i]
                desc = "\n".join(f"- {t}" for t in texts[:10])
                prompt = f"Graph context:\n{desc}\n\nQuestion: {q}\n\nAnswer:"
            else:
                num_nodes = int((batch == b).sum().item())
                prompt = f"Graph context: {num_nodes} nodes.\n\nQuestion: {q}\n\nAnswer:"
            augmented_prompts.append(prompt)

        # Tokenize augmented prompts
        enc = self.tokenizer(
            augmented_prompts,
            truncation=True,
            max_length=self.config['max_length'],
            padding='max_length',
            return_tensors='pt',
        )
        aug_input_ids = enc['input_ids'].to(device)
        aug_attention_mask = enc['attention_mask'].to(device)

        text_emb = self.llm.get_input_embeddings()(aug_input_ids)
        inputs_embeds = torch.cat([graph_emb, text_emb], dim=1)

        graph_mask = torch.ones(
            (aug_attention_mask.size(0), 1),
            device=device,
            dtype=aug_attention_mask.dtype,
        )
        aug_attention_mask = torch.cat([graph_mask, aug_attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=aug_attention_mask,
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
        pred_list = [p.strip().lower() for p in pred.split('|')][:k]
        gt_list = [g.strip().lower() for g in gt.split('|')]

        if any(g in ' '.join(pred_list) for g in gt_list):
            hits += 1

    return hits / len(predictions) * 100


def compute_recall_at_20(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Recall@20: fraction of ground-truth answers recalled within
    the first 20 whitespace tokens of the prediction.
    """
    recalls = []
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = pred.lower().split()[:20]
        gt_parts = [g.strip().lower() for g in gt.split('|')]

        recalled = sum(
            1 for g in gt_parts
            if any(g in tok or tok in g for tok in pred_tokens)
        )
        recalls.append(recalled / max(len(gt_parts), 1))

    return float(np.mean(recalls)) * 100


def compute_mrr(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Mean Reciprocal Rank: reciprocal of the 1-based token position
    where the first ground-truth match is found in the prediction.
    Returns 0 for a query if no match is found.
    """
    rr_scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = pred.lower().split()
        gt_parts = [g.strip().lower() for g in gt.split('|')]

        rr = 0.0
        for rank, tok in enumerate(pred_tokens, start=1):
            if any(g in tok or tok in g for g in gt_parts):
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return float(np.mean(rr_scores)) * 100


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
def evaluate(model, dataloader, config, mode: str = 'g_retriever'):
    """
    Evaluate model on `dataloader`.

    mode choices:
      - 'baseline'        : LLM only, no GNN
      - 'subgraph_pruning': prune to top-50% nodes by degree, then GNN+LLM
      - 'g_retriever'     : full GNN+LLM (default)
      - 'pipeline'        : GNN+LLM with node descriptions added to prompt

    Returns metrics dict with Hits@1, Hits@5, Recall@20, MRR (all as %).
    """
    model.eval()

    all_preds = []
    all_gts = []

    for batch_data in tqdm(dataloader, desc=f"Evaluating [{mode}]"):
        if mode == 'baseline':
            outputs = model.generate_baseline(
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_new_tokens=config["max_new_tokens"],
            )

        elif mode == 'subgraph_pruning':
            pruned_x, pruned_edge_index, pruned_batch = prune_subgraph(
                batch_data['x'],
                batch_data['edge_index'],
                batch_data['batch'],
                keep_ratio=0.5,
            )
            outputs = model.generate(
                x=pruned_x,
                edge_index=pruned_edge_index,
                batch=pruned_batch,
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_new_tokens=config["max_new_tokens"],
            )

        elif mode == 'pipeline':
            outputs = model.generate_pipeline(
                x=batch_data['x'],
                edge_index=batch_data['edge_index'],
                batch=batch_data['batch'],
                questions=batch_data['questions'],
                node_texts=batch_data.get('node_texts'),
                max_new_tokens=config["max_new_tokens"],
            )

        else:  # g_retriever (default)
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

    metrics = {
        'hits@1':      compute_hits_at_k(all_preds, all_gts, k=1),
        'hits@5':      compute_hits_at_k(all_preds, all_gts, k=5),
        'recall@20':   compute_recall_at_20(all_preds, all_gts),
        'mrr':         compute_mrr(all_preds, all_gts),
    }

    return metrics, all_preds, all_gts


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results_table(results_by_method: Dict[str, Dict]):
    """Print a formatted comparison table of all evaluated methods."""
    ordered_methods = [
        "Pipeline",
        "G-Retriever (GAT)",
        "G-Retriever (Transformer)",
        "G-Retriever (GCN)",
        "Subgraph Pruning",
        "Baseline",
    ]

    col_method = 25
    print("\n" + "=" * 70)
    print(
        f"{'Method':<{col_method}}| "
        f"{'Hits@1':>6} | "
        f"{'Hits@5':>6} | "
        f"{'Recall@20':>9} | "
        f"{'MRR':>6}"
    )
    print("-" * col_method + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 11 + "|" + "-" * 7)

    for method in ordered_methods:
        if method in results_by_method:
            m = results_by_method[method]
            print(
                f"{method:<{col_method}}| "
                f"{m.get('hits@1', 0.0):>6.2f} | "
                f"{m.get('hits@5', 0.0):>6.2f} | "
                f"{m.get('recall@20', 0.0):>9.2f} | "
                f"{m.get('mrr', 0.0):>6.2f}"
            )
        else:
            print(
                f"{method:<{col_method}}| "
                f"{'N/A':>6} | "
                f"{'N/A':>6} | "
                f"{'N/A':>9} | "
                f"{'N/A':>6}"
            )

    print("=" * 70)


def evaluate_all_modes(model, test_loader, config, encoder_type: str) -> Dict[str, Dict]:
    """
    Run evaluation with all four modes and return a results dict keyed by
    the display name used in the results table.
    """
    mode_to_label = {
        'baseline':        'Baseline',
        'subgraph_pruning':'Subgraph Pruning',
        'g_retriever':     f'G-Retriever ({encoder_type.upper()})',
        'pipeline':        'Pipeline',
    }

    results = {}
    for mode, label in mode_to_label.items():
        print(f"\n--- Evaluating: {label} ---")
        metrics, _, _ = evaluate(model, test_loader, config, mode=mode)
        results[label] = metrics
        print(f"  Hits@1={metrics['hits@1']:.2f}  Hits@5={metrics['hits@5']:.2f}  "
              f"Recall@20={metrics['recall@20']:.2f}  MRR={metrics['mrr']:.2f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train G-Retriever with GAT, Transformer, or GCN encoder")

    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed/train',
                        help='Directory containing pre-computed subgraphs')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')

    # Model
    parser.add_argument('--encoder', type=str, default='gat',
                        choices=['gat', 'transformer', 'gcn'],
                        help='GNN encoder type')
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='LLM model name')

    # Mode
    parser.add_argument('--mode', type=str, default='g_retriever',
                        choices=['baseline', 'subgraph_pruning', 'g_retriever', 'pipeline'],
                        help=(
                            'Evaluation/inference mode: '
                            'baseline=LLM only; '
                            'subgraph_pruning=prune top-50%% nodes then GNN+LLM; '
                            'g_retriever=full GNN+LLM; '
                            'pipeline=GNN+LLM with node descriptions in prompt'
                        ))

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
    print(f"  Mode:    {args.mode}")
    print(f"  Seed:    {args.seed}")
    print(f"  Output:  {output_dir}")
    print("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model (always create GRetriever; baseline mode just skips GNN at inference)
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
        print("\nRunning evaluation across all modes...")
        all_results = evaluate_all_modes(model, test_loader, config, args.encoder)

        # Also load saved results for other encoder types, if available
        for other_enc in ['gat', 'transformer', 'gcn']:
            if other_enc == args.encoder:
                continue
            label = f'G-Retriever ({other_enc.upper()})'
            results_path = os.path.join(args.output_dir, f"{other_enc}_seed{args.seed}", 'results.json')
            if os.path.exists(results_path):
                with open(results_path) as f:
                    saved = json.load(f)
                saved_metrics = saved.get('test_metrics', {})
                if saved_metrics:
                    all_results[label] = saved_metrics
                    print(f"Loaded saved results for {label}")

        print_results_table(all_results)
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

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, config)
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics, _, _ = evaluate(model, val_loader, config, mode=args.mode)
        print(
            f"Val  Hits@1={val_metrics['hits@1']:.2f}  "
            f"Hits@5={val_metrics['hits@5']:.2f}  "
            f"Recall@20={val_metrics['recall@20']:.2f}  "
            f"MRR={val_metrics['mrr']:.2f}"
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
        })

        # Save best model (by Hits@1)
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

        gc.collect()
        torch.cuda.empty_cache()

    # Final test evaluation — run all modes for the full comparison table
    print("\n" + "="*60)
    print("Final Test Evaluation (all modes)")
    print("="*60)

    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    all_results = evaluate_all_modes(model, test_loader, config, args.encoder)

    # Load saved results from other encoder runs if they exist
    for other_enc in ['gat', 'transformer', 'gcn']:
        if other_enc == args.encoder:
            continue
        label = f'G-Retriever ({other_enc.upper()})'
        results_path = os.path.join(args.output_dir, f"{other_enc}_seed{args.seed}", 'results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                saved = json.load(f)
            saved_metrics = saved.get('test_metrics', {})
            if saved_metrics:
                all_results[label] = saved_metrics
                print(f"Loaded saved results for {label}")

    print_results_table(all_results)

    # Use the trained mode's g_retriever metrics as the canonical test_metrics
    test_metrics = all_results.get(f'G-Retriever ({args.encoder.upper()})', {})

    # Save final results
    results = {
        'encoder': args.encoder,
        'mode': args.mode,
        'seed': args.seed,
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'all_mode_results': all_results,
        'best_val_hits@1': best_metric,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions from the primary mode
    _, test_preds, test_gts = evaluate(model, test_loader, config, mode=args.mode)
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump({
            'predictions': test_preds[:100],
            'ground_truths': test_gts[:100],
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
