"""
G-Retriever Training Script
============================
Architecture aligned with NVIDIA reference implementation.
Thesis contribution: Graph Transformer and GCN encoders in addition to GAT.

Usage:
    python train.py --encoder gat --epochs 2
    python train.py --encoder transformer --epochs 2
    python train.py --encoder gcn --epochs 2
    python train.py --encoder gat --eval_only --checkpoint results/gat_seed42/best_model.pt
"""

import os
import re
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, TransformerConv, GCNConv, global_mean_pool

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =============================================================================
# CONFIGURATION  (matches reference hyperparameters)
# =============================================================================

DEFAULT_CONFIG = {
    # GNN  — same dimensions as reference (1536 throughout)
    "gnn_in_channels":  1536,
    "gnn_hidden":       1536,
    "gnn_out":          1536,
    "gnn_layers":       4,
    "gnn_heads":        4,
    "gnn_dropout":      0.0,

    # LLM
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    "load_in_4bit": True,

    # LoRA
    "lora_r":       8,
    "lora_alpha":   16,
    "lora_dropout": 0.05,

    # Training  — matches reference
    "epochs":        2,
    "batch_size":    4,
    "eval_batch_size": 16,
    "lr":            1e-5,
    "weight_decay":  0.05,
    "max_grad_norm": 0.1,
    "grad_steps":    2,       # gradient accumulation steps
    "min_lr":        5e-6,
    "warmup_epochs": 1,

    # Generation
    "max_new_tokens": 32,
    "max_txt_len":    512,
}


# =============================================================================
# DATASET
# =============================================================================

class SubgraphDataset(torch.utils.data.Dataset):
    """Loads pre-computed .pt subgraph files."""

    def __init__(self, data_dir: str, split: str = 'all'):
        self.data_dir = data_dir
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

        # 80 / 10 / 10 split (deterministic by sorted filename order)
        n = len(files)
        train_end = int(0.8 * n)
        val_end   = int(0.9 * n)

        if split == 'train':
            self.files = files[:train_end]
        elif split == 'val':
            self.files = files[train_end:val_end]
        elif split == 'test':
            self.files = files[val_end:]
        else:
            self.files = files

        print(f"[{split}] {len(self.files)} subgraphs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.data_dir, self.files[idx]),
            weights_only=False,
        )
        # Compatibility: old files use 'answer' (int IDs), new use 'label' (names)
        if not hasattr(data, 'label') or not data.label:
            ans = getattr(data, 'answer', [])
            data.label = ' | '.join(str(a) for a in ans) if ans else ''
        if not hasattr(data, 'desc') or data.desc is None:
            data.desc = ''
        return data


# =============================================================================
# GNN ENCODERS
# =============================================================================

class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder using TransformerConv.
    Thesis contribution — transformer-style global attention over the graph.
    """

    def __init__(self, in_channels=1536, hidden_channels=1536,
                 out_channels=1536, num_layers=4, heads=4, dropout=0.0, **kwargs):
        super().__init__()
        head_dim = hidden_channels // heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden_channels
            is_last = (i == num_layers - 1)
            if is_last:
                self.convs.append(TransformerConv(
                    in_ch, out_channels, heads=1, concat=False,
                    dropout=dropout, beta=True))
                self.norms.append(nn.LayerNorm(out_channels))
            else:
                self.convs.append(TransformerConv(
                    in_ch, head_dim, heads=heads, concat=True,
                    dropout=dropout, beta=True))
                self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class GCNEncoder(nn.Module):
    """
    GCN encoder using GCNConv (Kipf & Welling 2017).
    Thesis contribution — spectral-based convolution for comparison.
    """

    def __init__(self, in_channels=1536, hidden_channels=1536,
                 out_channels=1536, num_layers=4, dropout=0.0, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))
            self.norms.append(nn.LayerNorm(out_ch))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class GATEncoder(nn.Module):
    """
    GAT encoder — matches reference architecture exactly.
    Uses GATConv with 4 heads and 1536-dim hidden/output.
    No LayerNorm between layers (matches PyG's built-in GAT).
    """

    def __init__(self, in_channels=1536, hidden_channels=1536,
                 out_channels=1536, num_layers=4, heads=4, dropout=0.0, **kwargs):
        super().__init__()
        head_dim = hidden_channels // heads
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_ch   = in_channels if i == 0 else hidden_channels
            is_last = (i == num_layers - 1)
            if is_last:
                self.convs.append(GATConv(
                    in_ch, out_channels, heads=1, concat=False,
                    dropout=dropout))
            else:
                self.convs.append(GATConv(
                    in_ch, head_dim, heads=heads, concat=True,
                    dropout=dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        return x


def get_gnn(encoder_type: str, config: dict) -> nn.Module:
    kwargs = dict(
        in_channels=config["gnn_in_channels"],
        hidden_channels=config["gnn_hidden"],
        out_channels=config["gnn_out"],
        num_layers=config["gnn_layers"],
        heads=config["gnn_heads"],
        dropout=config["gnn_dropout"],
    )
    if encoder_type == "gat":
        return GATEncoder(**kwargs)
    elif encoder_type == "transformer":
        return GraphTransformerEncoder(**kwargs)
    elif encoder_type == "gcn":
        return GCNEncoder(**kwargs)
    raise ValueError(f"Unknown encoder: {encoder_type}")


# =============================================================================
# SUBGRAPH PRUNING  (thesis mode)
# =============================================================================

def prune_subgraph(x, edge_index, batch, keep_ratio=0.5):
    """Keep top-50% nodes by degree per graph."""
    num_nodes = x.size(0)
    device = x.device

    degrees = torch.zeros(num_nodes, device=device)
    if edge_index.size(1) > 0:
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))

    keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    for b in batch.unique():
        idx = torch.where(batch == b)[0]
        n_keep = max(1, int(len(idx) * keep_ratio))
        _, top = torch.topk(degrees[idx], n_keep)
        keep_mask[idx[top]] = True

    node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    node_map[keep_mask] = torch.arange(keep_mask.sum(), device=device)

    new_x     = x[keep_mask]
    new_batch = batch[keep_mask]

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
    G-Retriever: GNN encoder + Llama-3.1-8B with LoRA.
    Interface matches the NVIDIA reference implementation.
    """

    def __init__(self, encoder_type: str = "gat", config: dict = None):
        super().__init__()
        config = config or DEFAULT_CONFIG
        self.config = config
        self.encoder_type = encoder_type

        # ── GNN ──────────────────────────────────────────────────────────────
        print(f"Creating {encoder_type.upper()} encoder...")
        self.gnn = get_gnn(encoder_type, config)

        # ── LLM (4-bit quantised) ────────────────────────────────────────────
        print(f"Loading LLM: {config['llm_name']} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if config["load_in_4bit"] else None

        self.tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.llm = AutoModelForCausalLM.from_pretrained(
            config['llm_name'],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # ── LoRA ─────────────────────────────────────────────────────────────
        print("Applying LoRA...")
        self.llm = prepare_model_for_kbit_training(self.llm)
        lora_cfg = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm.print_trainable_parameters()

        # ── Graph → LLM projection  (matches PyG GRetriever exactly) ────────
        llm_hidden = self.llm.config.hidden_size
        self.graph_proj = nn.Sequential(
            nn.Linear(config["gnn_out"], llm_hidden),
            nn.Sigmoid(),
            nn.Linear(llm_hidden, llm_hidden),
        )

        # Move GNN and projection to same device as LLM
        lm_device = next(iter(self.llm.parameters())).device
        self.graph_proj = self.graph_proj.to(lm_device)
        self.gnn        = self.gnn.to(lm_device)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_graph(self, x, edge_index, batch):
        """GNN → pool → project → cast to LLM dtype → [B, 1, H]."""
        node_emb  = self.gnn(x, edge_index)                          # [N, gnn_out]
        graph_emb = global_mean_pool(node_emb, batch)                 # [B, gnn_out]
        graph_emb = self.graph_proj(graph_emb.float())                # [B, llm_hidden]
        dtype     = self.llm.get_input_embeddings().weight.dtype
        return graph_emb.to(dtype).unsqueeze(1)                       # [B, 1, llm_hidden]

    def _build_prompts(self, questions, desc, include_desc=False):
        prompts = []
        for q, d in zip(questions, desc):
            if include_desc and d:
                prompts.append(f"Context:\n{d}\n\nQuestion: {q}\n\nAnswer:")
            else:
                prompts.append(f"Question: {q}\n\nAnswer:")
        return prompts

    def _tokenize(self, texts, max_length=None):
        max_length = max_length or self.config["max_txt_len"]
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )

    # ── Training forward (always g_retriever mode) ────────────────────────────

    def forward(self, questions, x, edge_index, batch, labels,
                edge_attr=None, desc=None):
        """Compute cross-entropy loss. Called during training."""
        device = next(iter(self.llm.parameters())).device
        x          = x.to(device)
        edge_index = edge_index.to(device)
        batch      = batch.to(device)

        desc = desc or [''] * len(questions)

        # Build full text: prompt + answer
        prompts = self._build_prompts(questions, desc, include_desc=False)
        full_texts = [p + ' ' + l for p, l in zip(prompts, labels)]

        prompt_enc = self._tokenize(prompts)
        full_enc   = self._tokenize(full_texts)

        input_ids      = full_enc['input_ids'].to(device)
        attention_mask = full_enc['attention_mask'].to(device)

        # Labels: mask prompt tokens with -100
        lm_labels = input_ids.clone()
        for i, pmask in enumerate(prompt_enc['attention_mask']):
            prompt_len = int(pmask.sum().item())
            lm_labels[i, :prompt_len] = -100

        # Encode graph and prepend to embeddings
        graph_emb  = self._encode_graph(x, edge_index, batch)         # [B, 1, H]
        text_emb   = self.llm.get_input_embeddings()(input_ids)        # [B, L, H]
        inputs_emb = torch.cat([graph_emb, text_emb], dim=1)          # [B, 1+L, H]

        graph_mask     = torch.ones(graph_emb.size(0), 1, device=device,
                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat([graph_mask, attention_mask], dim=1)

        label_pad = torch.full((lm_labels.size(0), 1), -100,
                               device=device, dtype=lm_labels.dtype)
        lm_labels = torch.cat([label_pad, lm_labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_emb,
            attention_mask=attention_mask,
            labels=lm_labels,
            return_dict=True,
        )
        return outputs.loss

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(self, questions, x, edge_index, batch,
                  edge_attr=None, desc=None, mode='g_retriever'):
        """
        Generate predictions. Returns list of prediction strings.

        mode:
          'g_retriever'     – full GNN + LLM  (default, matches reference)
          'subgraph_pruning' – prune top-50% nodes then GNN + LLM
          'pipeline'         – GNN + LLM with desc injected in prompt
          'baseline'         – LLM only, no GNN (uses desc as context)
        """
        device = next(iter(self.llm.parameters())).device
        desc   = desc or [''] * len(questions)

        # Use left-padding for inference so real tokens are at the right
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        include_desc = (mode in ('baseline', 'pipeline'))
        prompts      = self._build_prompts(questions, desc, include_desc)
        enc          = self._tokenize(prompts)
        input_ids    = enc['input_ids'].to(device)
        attn_mask    = enc['attention_mask'].to(device)

        if mode == 'baseline':
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.config['max_new_tokens'],
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
            # Decode only newly generated tokens
            gen_tokens = outputs[:, input_ids.size(1):]

        else:
            x          = x.to(device)
            edge_index = edge_index.to(device)
            batch      = batch.to(device)

            if mode == 'subgraph_pruning':
                x, edge_index, batch = prune_subgraph(x, edge_index, batch)

            graph_emb  = self._encode_graph(x, edge_index, batch)
            text_emb   = self.llm.get_input_embeddings()(input_ids)
            inputs_emb = torch.cat([graph_emb, text_emb], dim=1)
            input_len  = inputs_emb.size(1)

            graph_mask = torch.ones(graph_emb.size(0), 1, device=device,
                                    dtype=attn_mask.dtype)
            attn_mask  = torch.cat([graph_mask, attn_mask], dim=1)

            outputs = self.llm.generate(
                inputs_embeds=inputs_emb,
                attention_mask=attn_mask,
                max_new_tokens=self.config['max_new_tokens'],
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
            # When using inputs_embeds, some Transformers versions return only
            # new tokens; others return full sequence. Handle both cases:
            if outputs.size(1) > self.config['max_new_tokens']:
                gen_tokens = outputs[:, input_len:]
            else:
                gen_tokens = outputs

        self.tokenizer.padding_side = orig_padding_side
        preds = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return [p.strip() for p in preds]


# =============================================================================
# METRICS  (aligned with reference compute_metrics)
# =============================================================================

def compute_metrics(predictions, ground_truths):
    """
    Evaluate predictions against ground truth labels.
    Both are lists of '|'-separated entity name strings.
    Returns dict of all metrics (as fractions, not percentages).
    """
    all_hit          = []
    all_hit1         = []
    all_hit5         = []
    all_hit_any      = []
    all_precision    = []
    all_recall       = []
    all_recall_at_20 = []
    all_rr           = []
    all_f1           = []

    for pred_str, label_str in zip(predictions, ground_truths):
        # Split on '|' to get ranked candidate list (matches reference)
        pred  = [p.strip().lower() for p in pred_str.split('|') if p.strip()]
        label = [l.strip().lower() for l in label_str.split('|') if l.strip()]

        if not pred or not label:
            for lst in [all_hit, all_hit1, all_hit5, all_hit_any,
                        all_precision, all_recall, all_recall_at_20, all_rr, all_f1]:
                lst.append(0)
            continue

        # Substring hit@1 (reference style: regex find)
        try:
            hit = len(re.findall(re.escape(pred[0]), ' | '.join(label))) > 0
        except Exception:
            hit = False
        all_hit.append(int(hit))

        matches        = set(pred) & set(label)
        matches_at_20  = set(pred[:20]) & set(label)

        precision     = len(matches)       / len(set(pred))
        recall        = len(matches)       / len(set(label))
        recall_at_20  = len(matches_at_20) / len(set(label))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        hit1  = int(pred[0] in label)
        hit5  = int(len(set(pred[:5]) & set(label)) > 0)

        rr = 0.0
        for rank, p in enumerate(pred, start=1):
            if p in label:
                rr = 1.0 / rank
                break

        all_hit1.append(hit1)
        all_hit5.append(hit5)
        all_hit_any.append(int(precision > 0))
        all_precision.append(precision)
        all_recall.append(recall)
        all_recall_at_20.append(recall_at_20)
        all_rr.append(rr)
        all_f1.append(f1)

    n = len(predictions)
    return {
        'hit@1':       sum(all_hit)          / n,
        'exact_hit@1': sum(all_hit1)         / n,
        'hit@5':       sum(all_hit5)         / n,
        'hit@any':     sum(all_hit_any)      / n,
        'precision':   sum(all_precision)    / n,
        'recall':      sum(all_recall)       / n,
        'recall@20':   sum(all_recall_at_20) / n,
        'mrr':         sum(all_rr)           / n,
        'f1':          sum(all_f1)           / n,
    }


def print_metrics(metrics: dict, prefix: str = ''):
    print(f"{prefix}F1:              {metrics['f1']:.4f}")
    print(f"{prefix}Precision:       {metrics['precision']:.4f}")
    print(f"{prefix}Recall:          {metrics['recall']:.4f}")
    print(f"{prefix}Substring hit@1: {metrics['hit@1']:.4f}")
    print(f"{prefix}Exact hit@1:     {metrics['exact_hit@1']:.4f}")
    print(f"{prefix}Hit@5:           {metrics['hit@5']:.4f}")
    print(f"{prefix}Hit@any:         {metrics['hit@any']:.4f}")
    print(f"{prefix}Recall@20:       {metrics['recall@20']:.4f}")
    print(f"{prefix}MRR:             {metrics['mrr']:.4f}")


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results_table(results: dict):
    order = [
        "Pipeline",
        f"G-Retriever (GAT)",
        f"G-Retriever (Transformer)",
        f"G-Retriever (GCN)",
        "Subgraph Pruning",
        "Baseline (LLM only)",
    ]
    W = 26
    print("\n" + "=" * 78)
    print(f"{'Method':<{W}}| {'Hit@1':>7} | {'Hit@5':>7} | {'Recall@20':>9} | {'MRR':>7} | {'F1':>7}")
    print("-" * W + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 9 + "|" + "-" * 8)
    for method in order:
        if method in results:
            m = results[method]
            print(f"{method:<{W}}| "
                  f"{m['exact_hit@1']*100:>6.2f}% | "
                  f"{m['hit@5']*100:>6.2f}% | "
                  f"{m['recall@20']*100:>8.2f}% | "
                  f"{m['mrr']*100:>6.2f}% | "
                  f"{m['f1']*100:>6.2f}%")
        else:
            print(f"{method:<{W}}| {'N/A':>7} | {'N/A':>7} | {'N/A':>9} | {'N/A':>7} | {'N/A':>7}")
    print("=" * 78)


# =============================================================================
# LR SCHEDULE  (cosine with warmup — exact reference implementation)
# =============================================================================

def adjust_learning_rate(param_group, lr_base, epoch, num_epochs,
                         warmup_epochs=1, min_lr=5e-6):
    if epoch < warmup_epochs:
        lr = lr_base
    else:
        lr = min_lr + (lr_base - min_lr) * 0.5 * (
            1.0 + math.cos(
                math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            )
        )
    param_group['lr'] = lr
    return lr


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, epoch, config):
    model.train()
    total_loss = 0
    num_batches = 0
    grad_steps = config['grad_steps']

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        loss = model(
            questions  = batch.question,
            x          = batch.x,
            edge_index = batch.edge_index,
            batch      = batch.batch,
            labels     = batch.label,
            edge_attr  = getattr(batch, 'edge_attr', None),
            desc       = batch.desc if hasattr(batch, 'desc') else None,
        )

        loss.backward()

        if (step + 1) % grad_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=config['max_grad_norm'],
            )
            adjust_learning_rate(
                optimizer.param_groups[0],
                lr_base=config['lr'],
                epoch=step / len(loader) + epoch,
                num_epochs=config['epochs'],
                warmup_epochs=config['warmup_epochs'],
                min_lr=config['min_lr'],
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss  += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'lr':   f'{optimizer.param_groups[0]["lr"]:.2e}'})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_loader(model, loader, config, mode='g_retriever', print_n=3):
    model.eval()
    all_preds = []
    all_gts   = []
    printed   = 0

    for batch in tqdm(loader, desc=f"Eval [{mode}]"):
        preds = model.inference(
            questions  = batch.question,
            x          = batch.x,
            edge_index = batch.edge_index,
            batch      = batch.batch,
            edge_attr  = getattr(batch, 'edge_attr', None),
            desc       = batch.desc if hasattr(batch, 'desc') else None,
            mode       = mode,
        )
        all_preds.extend(preds)
        all_gts.extend(batch.label)

        # Print first few predictions for debugging
        if printed < print_n:
            for p, l, q in zip(preds, batch.label, batch.question):
                if printed >= print_n:
                    break
                print(f"\n  [DEBUG {mode}] Q: {q[:80]}")
                print(f"  [DEBUG {mode}] Pred:  {repr(p[:120])}")
                print(f"  [DEBUG {mode}] Label: {repr(l[:120])}")
                printed += 1

    return compute_metrics(all_preds, all_gts), all_preds, all_gts


def evaluate_all_modes(model, loader, config, encoder_type):
    """Run all 4 modes and return results dict keyed by display name."""
    modes = {
        'baseline':         'Baseline (LLM only)',
        'subgraph_pruning': 'Subgraph Pruning',
        'g_retriever':      f'G-Retriever ({encoder_type.upper()})',
        'pipeline':         'Pipeline',
    }
    results = {}
    for mode, label in modes.items():
        print(f"\n--- {label} ---")
        metrics, _, _ = evaluate_loader(model, loader, config, mode=mode)
        results[label] = metrics
        print_metrics(metrics, prefix='  ')
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--encoder',    type=str, default='gat',
                        choices=['gat', 'transformer', 'gcn'])
    parser.add_argument('--llm',        type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--epochs',     type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--eval_only',  action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config['llm_name'] = args.llm
    if args.epochs     is not None: config['epochs']     = args.epochs
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.lr         is not None: config['lr']         = args.lr

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = os.path.join(args.output_dir, f"{args.encoder}_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"G-Retriever  |  Encoder: {args.encoder.upper()}  |  Seed: {args.seed}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print("=" * 60)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = SubgraphDataset(args.data_dir, split='train')
    val_ds   = SubgraphDataset(args.data_dir, split='val')
    test_ds  = SubgraphDataset(args.data_dir, split='test')

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  drop_last=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['eval_batch_size'],
                              shuffle=False, drop_last=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config['eval_batch_size'],
                              shuffle=False, drop_last=False, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nCreating model...")
    model = GRetriever(encoder_type=args.encoder, config=config)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # ── Eval only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        all_results = evaluate_all_modes(model, test_loader, config, args.encoder)
        # Merge saved results from other encoders if available
        for enc in ['gat', 'transformer', 'gcn']:
            if enc == args.encoder:
                continue
            rpath = os.path.join(args.output_dir, f"{enc}_seed{args.seed}", 'results.json')
            if os.path.exists(rpath):
                with open(rpath) as f:
                    saved = json.load(f)
                label = f'G-Retriever ({enc.upper()})'
                if saved.get('test_metrics'):
                    all_results[label] = saved['test_metrics']
        print_results_table(all_results)
        return

    # ── Optimizer ─────────────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': config['lr'], 'weight_decay': config['weight_decay']}],
        betas=(0.9, 0.95),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = []
    start_time = time.time()

    for epoch in range(config['epochs']):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{config['epochs']}\n{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, epoch, config)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation (loss only — fast)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val loss"):
                loss = model(
                    questions  = batch.question,
                    x          = batch.x,
                    edge_index = batch.edge_index,
                    batch      = batch.batch,
                    labels     = batch.label,
                    edge_attr  = getattr(batch, 'edge_attr', None),
                    desc       = batch.desc if hasattr(batch, 'desc') else None,
                )
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Val   Loss: {val_loss:.4f}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"*** Saved best model (val_loss={best_val_loss:.4f}) ***")

        torch.cuda.empty_cache()

    print(f"\nTotal training time: {(time.time()-start_time)/60:.1f} min")

    # ── Final test evaluation (all 4 modes) ───────────────────────────────────
    print("\n" + "="*60)
    print("Final Test Evaluation — loading best checkpoint")
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    all_results = evaluate_all_modes(model, test_loader, config, args.encoder)

    # Merge other encoder results if available
    for enc in ['gat', 'transformer', 'gcn']:
        if enc == args.encoder:
            continue
        rpath = os.path.join(args.output_dir, f"{enc}_seed{args.seed}", 'results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                saved = json.load(f)
            label = f'G-Retriever ({enc.upper()})'
            if saved.get('test_metrics'):
                all_results[label] = saved['test_metrics']
                print(f"Loaded saved results for {label}")

    print_results_table(all_results)

    # Save
    test_metrics = all_results.get(f'G-Retriever ({args.encoder.upper()})', {})
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'encoder':      args.encoder,
            'seed':         args.seed,
            'config':       config,
            'history':      history,
            'test_metrics': test_metrics,
            'all_results':  {k: v for k, v in all_results.items()},
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
