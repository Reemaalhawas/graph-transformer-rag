import argparse
import json
import math
import os
import time

import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader

from stark_qa import load_qa
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric import seed_everything
from torch_geometric.nn import GRetriever
from torch_geometric.nn.nlp import LLM
from tqdm import tqdm

from compute_metrics import compute_metrics

from STaRKQADataset import STaRKQADataset
from STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset

from torch_geometric.nn import GAT
from src.models import build_gcn, GraphTransformerEncoder, CombinedEncoder

RESULTS_DIR = "qa_results"
FS_CKPT = "/home/ubuntu/GraphRAG/checkpoints"


# ─────────────────────────────────────────────────────────── GNN factory ──


def build_gnn(encoder, in_channels, hidden_channels, out_channels, num_layers, heads):
    if encoder == 'gat':
        return GAT(in_channels=in_channels, hidden_channels=hidden_channels,
                   out_channels=out_channels, num_layers=num_layers, heads=heads)
    elif encoder == 'gcn':
        return build_gcn(in_channels=in_channels, hidden_channels=hidden_channels,
                         out_channels=out_channels, num_layers=num_layers)
    elif encoder == 'graph_transformer':
        return GraphTransformerEncoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                       out_channels=out_channels, num_layers=num_layers, heads=heads)
    elif encoder == 'combined':
        return CombinedEncoder(in_channels=in_channels, hidden_channels=hidden_channels,
                               out_channels=out_channels, num_layers=num_layers, heads=heads)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose: gat, gcn, graph_transformer, combined")


# ───────────────────────────────────────────────── loss / inference helpers ──


def get_loss(model, batch, mode) -> Tensor:
    if mode == 'pcst_only':
        return model(batch.question, batch.label, batch.desc)
    else:
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc)


def inference_step(model, batch, mode):
    if mode == 'pcst_only':
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc)


# ──────────────────────────────────────────────── checkpoint helpers ──


def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {k: v.requires_grad for k, v in model.named_parameters()}
    for k in list(state_dict.keys()):
        if k in param_grad_dict and not param_grad_dict[k]:
            del state_dict[k]
    torch.save(state_dict, save_path)
    os.makedirs(FS_CKPT, exist_ok=True)
    fs_path = os.path.join(FS_CKPT, os.path.basename(save_path))
    torch.save(state_dict, fs_path)
    print(f"Checkpoint saved to {fs_path}")


def load_params_dict(model, save_path):
    state_dict = model.state_dict()
    state_dict.update(torch.load(save_path, weights_only=True))
    model.load_state_dict(state_dict)
    return model


# ──────────────────────────────────────────── results saving & printing ──


def save_results(model_save_name: str, mode: str, encoder: str, metrics: dict):
    """Persist metrics as JSON, then print the full comparison table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {'model': model_save_name, 'mode': mode, 'encoder': encoder, **metrics}
    path = os.path.join(RESULTS_DIR, f"{model_save_name}_results.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {path}")
    _print_results_table()


def _print_results_table():
    """Load every saved JSON and print a side-by-side comparison table."""
    if not os.path.isdir(RESULTS_DIR):
        return
    rows = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('_results.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                rows.append(json.load(f))
    if not rows:
        return

    metric_keys = [k for k in rows[0] if k not in ('model', 'mode', 'encoder')]

    sep = "=" * 110
    print(f"\n{sep}")
    print("  RESULTS COMPARISON  (all completed experiments)")
    print(sep)
    header = f"{'Model':<38} {'Mode':<12} {'Encoder':<18}"
    for k in metric_keys:
        header += f"  {k[:10]:>10}"
    print(header)
    print("-" * 110)
    for row in rows:
        line = f"{row.get('model',''):<38} {row.get('mode',''):<12} {row.get('encoder',''):<18}"
        for k in metric_keys:
            v = row.get(k, '')
            line += f"  {v:>10.4f}" if isinstance(v, float) else f"  {str(v):>10}"
        print(line)
    print(sep)


# ──────────────────────────────────────────── baseline evaluation ──


def _eval_baseline(qa_raw_test, retrieval_config_version: int) -> dict:
    """
    Baseline: pure vector search — no GNN, no LLM.
    Evaluates whether the correct answer names appear in the top-k retrieved
    node descriptions returned by vector similarity search.
    """
    print("\n=== Baseline: Pure Vector Search (no GNN, no LLM) ===")
    root_path = f"stark_qa_vector_rag_{retrieval_config_version}"
    test_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_test, split="test")

    hits, hits_at_5, precisions, recalls, reciprocal_ranks = [], [], [], [], []

    for data in tqdm(test_dataset, desc="Baseline eval"):
        retrieved_text = data.desc.lower()
        answers = [a.strip() for a in data.label.split('|') if a.strip()]

        matched = [a for a in answers if a in retrieved_text]
        n_matched = len(matched)

        hits.append(float(n_matched > 0))
        precisions.append(n_matched / max(len(answers), 1))
        recalls.append(n_matched / max(len(answers), 1))

        # Hit@5: look only at the first 5 data rows (skip CSV header)
        lines = retrieved_text.split('\n')
        top5_text = '\n'.join(lines[:6])
        hits_at_5.append(float(any(a in top5_text for a in answers)))

        # MRR
        rr = 0.0
        for i, line in enumerate(lines[1:7], start=1):
            if any(a in line for a in answers):
                rr = 1.0 / i
                break
        reciprocal_ranks.append(rr)

    metrics = {
        'hit_rate':  float(np.mean(hits)),
        'hit_at_5':  float(np.mean(hits_at_5)),
        'precision': float(np.mean(precisions)),
        'recall':    float(np.mean(recalls)),
        'mrr':       float(np.mean(reciprocal_ranks)),
    }
    print(f"  Hit Rate:  {metrics['hit_rate']:.4f}")
    print(f"  Hit@5:     {metrics['hit_at_5']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    return metrics


# ──────────────────────────────────────────────────────────── main train ──


def train(
    num_epochs,
    hidden_channels,
    num_gnn_layers,
    batch_size,
    eval_batch_size,
    lr,
    llama_version,
    retrieval_config_version,
    algo_config_version,
    g_retriever_config_version,
    encoder='gat',
    checkpointing=False,
    freeze_llm=False,
    num_gpus=None,
    mode='gretriever',
):
    def adjust_learning_rate(param_group, LR, epoch):
        min_lr = 5e-6
        warmup_epochs = 1
        if epoch < warmup_epochs:
            lrate = LR
        else:
            lrate = min_lr + (LR - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                               (num_epochs - warmup_epochs)))
        param_group['lr'] = lrate
        return lrate

    start_time = time.time()
    qa_dataset = load_qa("prime")
    qa_raw_train = qa_dataset.get_subset('train')
    qa_raw_val   = qa_dataset.get_subset('val')
    qa_raw_test  = qa_dataset.get_subset('test')
    seed_everything(42)

    # ── Baseline: no LLM, no GNN — just retrieval quality ──────────────────
    if mode == 'baseline':
        metrics = _eval_baseline(qa_raw_test, retrieval_config_version)
        save_results('vector-baseline', 'baseline', 'none', metrics)
        return

    # ── Determine paths & names ─────────────────────────────────────────────
    if mode == 'pcst_only':
        model_save_name = f'llm-pcst-{llama_version}'
        root_path       = f"stark_qa_v{retrieval_config_version}_{algo_config_version}"
    elif mode == 'gretriever':
        model_save_name = f'{encoder}-gretriever-{llama_version}'
        root_path       = f"stark_qa_v{retrieval_config_version}_{algo_config_version}"
    elif mode == 'pipeline':
        model_save_name = f'{encoder}-pipeline-{llama_version}'
        root_path       = f"stark_qa_pipeline_v{retrieval_config_version}_{algo_config_version}"
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: baseline, pcst_only, gretriever, pipeline")

    enc_label = 'none' if mode == 'pcst_only' else encoder
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}  |  Encoder: {enc_label}  |  LLM: {llama_version}")
    print(f"{'='*60}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    pipeline_mode = (mode == 'pipeline')
    print("Loading datasets...")
    t = time.time()
    train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version,
                                   algo_config_version, split="train",
                                   pipeline_mode=pipeline_mode)
    print(f"  train: {len(train_dataset)} samples  ({time.time()-t:.1f}s)")
    val_dataset = STaRKQADataset(root_path, qa_raw_val, retrieval_config_version,
                                 algo_config_version, split="val",
                                 pipeline_mode=pipeline_mode)
    test_dataset = STaRKQADataset(root_path, qa_raw_test, retrieval_config_version,
                                  algo_config_version, split="test",
                                  pipeline_mode=pipeline_mode)
    os.makedirs(f'{root_path}/models', exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=eval_batch_size,
                              drop_last=False, pin_memory=True, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=eval_batch_size,
                              drop_last=False, pin_memory=True, shuffle=False)

    # ── LLM ──────────────────────────────────────────────────────────────────
    if llama_version == 'tiny_llama':
        llm = LLM(model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1')
    elif llama_version == 'llama2-7b':
        llm = LLM(model_name='meta-llama/Llama-2-7b-chat-hf')
    elif llama_version == 'llama3.1-8b':
        llm = LLM(model_name='meta-llama/Llama-3.1-8B-Instruct')
    else:
        raise ValueError(f"Unknown llama_version: {llama_version}")

    if freeze_llm:
        for param in llm.parameters():
            param.requires_grad = False

    # ── GNN (only for gretriever / pipeline) ─────────────────────────────────
    if mode in ('gretriever', 'pipeline'):
        gnn = build_gnn(
            encoder=encoder,
            in_channels=1536,
            hidden_channels=hidden_channels,
            out_channels=1536,
            num_layers=num_gnn_layers,
            heads=4,
        )
        n_params = sum(p.numel() for p in gnn.parameters())
        print(f"GNN  encoder={encoder}  layers={num_gnn_layers}  "
              f"hidden={hidden_channels}  params={n_params:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if mode == 'pcst_only':
        model = llm
    else:
        if llama_version == 'tiny_llama':
            model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
        else:
            model = GRetriever(llm=llm, gnn=gnn)

    print(f"Model device: {llm.device}")

    params    = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': 0.05}],
                                  betas=(0.9, 0.95))
    grad_steps    = 2
    best_epoch    = 0
    best_val_loss = float('inf')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        if epoch == 0:
            print(f"Preparation time: {time.time() - start_time:.2f}s")
            start_time = time.time()
            print("Training started...")
        epoch_str = f'Epoch {epoch + 1}/{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = get_loss(model, batch, mode)
            loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % grad_steps == 0:
                lr = adjust_learning_rate(optimizer.param_groups[0], lr,
                                          step / len(train_loader) + epoch)
            optimizer.step()
            epoch_loss += float(loss)

        print(f"{epoch_str}  Train Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += get_loss(model, batch, mode).item()
        val_loss /= len(val_loader)
        print(f"{epoch_str}  Val Loss:   {val_loss:.4f}")

        if checkpointing and val_loss < best_val_loss:
            print("  → Checkpointing best model")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(model, f'{root_path}/models/{model_save_name}_best_ckpt.pt')

    if llm.device.type != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint for final eval...")
        model = load_params_dict(model, f'{root_path}/models/{model_save_name}_best_ckpt.pt')

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.eval()
    eval_output = []
    print("\nRunning final evaluation on test set...")
    for batch in tqdm(test_loader, desc="Eval"):
        with torch.no_grad():
            pred = inference_step(model, batch, mode)
            eval_output.append({
                'pred':     pred,
                'question': batch.question,
                'desc':     batch.desc,
                'label':    batch.label,
            })

    metrics = compute_metrics(eval_output)
    print(f"\nTotal training time: {time.time() - start_time:.2f}s")

    # Save model weights + raw predictions
    save_params_dict(model, f'{root_path}/models/{model_save_name}.pt')
    torch.save(eval_output, f'{root_path}/models/{model_save_name}_eval_outs.pt')

    # Save metrics JSON and print comparison table
    save_results(model_save_name, mode, enc_label, metrics)


# ──────────────────────────────────────────────────────────────── CLI ──


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Graph-RAG experiment runner — one call per experiment mode'
    )
    parser.add_argument(
        '--mode', type=str, default='gretriever',
        choices=['baseline', 'pcst_only', 'gretriever', 'pipeline'],
        help=(
            'baseline   = pure vector search, no GNN, no LLM\n'
            'pcst_only  = PCST subgraph → LLM (no GNN)\n'
            'gretriever = PCST subgraph → GNN → LLM\n'
            'pipeline   = PCST subgraph + extra base nodes → GNN → LLM'
        ),
    )
    parser.add_argument('--encoder', type=str, default='gat',
                        choices=['gat', 'gcn', 'graph_transformer', 'combined'])
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers',      type=int, default=4)
    parser.add_argument('--lr',                  type=float, default=1e-5)
    parser.add_argument('--epochs',              type=int, default=2)
    parser.add_argument('--batch_size',          type=int, default=4)
    parser.add_argument('--eval_batch_size',     type=int, default=16)
    parser.add_argument('--checkpointing',       action='store_true')
    parser.add_argument('--freeze_llm',          action='store_true')
    parser.add_argument('--llama_version',       type=str, default='llama3.1-8b',
                        choices=['tiny_llama', 'llama2-7b', 'llama3.1-8b'])
    parser.add_argument('--retrieval_config_version',  type=int, default=0)
    parser.add_argument('--algo_config_version',       type=int, default=0)
    parser.add_argument('--g_retriever_config_version', type=int, default=0)
    args = parser.parse_args()

    load_dotenv('db.env', override=True)

    t0 = time.time()
    train(
        num_epochs=args.epochs,
        hidden_channels=args.gnn_hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        llama_version=args.llama_version,
        retrieval_config_version=args.retrieval_config_version,
        algo_config_version=args.algo_config_version,
        g_retriever_config_version=args.g_retriever_config_version,
        encoder=args.encoder,
        checkpointing=args.checkpointing,
        freeze_llm=args.freeze_llm,
        mode=args.mode,
    )
    print(f"\nTotal wall time: {time.time() - t0:.2f}s")
