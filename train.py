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
from torch_geometric.nn import GAT, GCN, GRetriever
from torch_geometric.nn.nlp import LLM
from tqdm import tqdm
from compute_metrics import compute_metrics
from STaRKQADatasetGDS import STaRKQADataset
from STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset


def get_loss(model, batch, model_save_name) -> Tensor:
    if model_save_name.startswith('llm'):
        return model(batch.question, batch.label, batch.desc)
    else:
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc)


def inference_step(model, batch, model_save_name):
    if model_save_name.startswith('llm'):
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc)


def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]
    torch.save(state_dict, save_path)


def load_params_dict(model, save_path):
    state_dict = model.state_dict()
    state_dict.update(torch.load(save_path))
    model.load_state_dict(state_dict)
    return model


# =============================================================================
# METRICS
# =============================================================================

def _normalize(text):
    return text.strip().lower()


def compute_hits_at_k(predictions, ground_truths, k):
    hits = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_list = [_normalize(p) for p in pred.split('|')][:k]
        gt_list = [_normalize(g) for g in gt.split('|')]
        if any(g in ' '.join(pred_list) for g in gt_list):
            hits += 1
    return hits / len(predictions) * 100 if predictions else 0.0


def compute_recall_at_k(predictions, ground_truths, k=20):
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred_list = [_normalize(p) for p in pred.split('|')][:k]
        gt_list = [_normalize(g) for g in gt.split('|')]
        if not gt_list:
            scores.append(0.0)
            continue
        found = sum(1 for g in gt_list if any(g in p for p in pred_list))
        scores.append(found / len(gt_list))
    return sum(scores) / len(scores) * 100 if scores else 0.0


def compute_mrr(predictions, ground_truths):
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred_list = [_normalize(p) for p in pred.split('|')]
        gt_list = [_normalize(g) for g in gt.split('|')]
        rr = 0.0
        for rank, p in enumerate(pred_list, 1):
            if any(g in p for g in gt_list):
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) * 100 if scores else 0.0


def compute_paper_metrics(eval_output):
    all_preds = []
    all_labels = []
    for batch_data in eval_output:
        preds = batch_data['pred'] if isinstance(batch_data['pred'], list) else [batch_data['pred']]
        labels = batch_data['label'] if isinstance(batch_data['label'], list) else [batch_data['label']]
        all_preds.extend(preds)
        all_labels.extend(labels)

    metrics = {
        'hits@1':    compute_hits_at_k(all_preds, all_labels, k=1),
        'hits@5':    compute_hits_at_k(all_preds, all_labels, k=5),
        'recall@20': compute_recall_at_k(all_preds, all_labels, k=20),
        'mrr':       compute_mrr(all_preds, all_labels),
    }
    print("\n=== Paper Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}%")
    return metrics


# =============================================================================
# TRAINING
# =============================================================================

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
    sys_prompt=None,
    num_gpus=None,
    freeze_llm=False,
):
    def adjust_learning_rate(param_group, LR, epoch):
        min_lr = 5e-6
        warmup_epochs = 1
        if epoch < warmup_epochs:
            lr = LR
        else:
            lr = min_lr + (LR - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                               (num_epochs - warmup_epochs)))
        param_group['lr'] = lr
        return lr

    start_time = time.time()
    qa_dataset = load_qa("prime")
    qa_raw_train = qa_dataset.get_subset('train')
    qa_raw_val = qa_dataset.get_subset('val')
    qa_raw_test = qa_dataset.get_subset('test')
    seed_everything(42)

    if num_gnn_layers == 0:
        model_save_name = f'llm-{llama_version}'
    else:
        if freeze_llm:
            model_save_name = f'{encoder}-frozen-llm-{llama_version}'
        else:
            model_save_name = f'{encoder}-llm-{llama_version}'

    if model_save_name == f'llm-{llama_version}':
        root_path = f"stark_qa_vector_rag_{retrieval_config_version}"
        print("Loading stark-qa prime train dataset...")
        t = time.time()
        train_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_train, split="train")
        print(f'Finished loading train dataset in {time.time() - t:.2f}s')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_val, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_test, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)
    else:
        root_path = f"stark_qa_v{retrieval_config_version}_{algo_config_version}"
        print("Loading stark-qa prime train dataset...")
        t = time.time()
        train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version, algo_config_version, split="train")
        print(f'Finished loading train dataset in {time.time() - t:.2f}s')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQADataset(root_path, qa_raw_val, retrieval_config_version, algo_config_version, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQADataset(root_path, qa_raw_test, retrieval_config_version, algo_config_version, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # GNN encoder selection
    if encoder == 'gcn':
        gnn = GCN(
            in_channels=1536,
            hidden_channels=hidden_channels,
            out_channels=1536,
            num_layers=num_gnn_layers,
        )
    else:  # default: gat
        gnn = GAT(
            in_channels=1536,
            hidden_channels=hidden_channels,
            out_channels=1536,
            num_layers=num_gnn_layers,
            heads=4,
        )

    # LLM selection
    llm_model_map = {
        'tiny_llama': 'TinyLlama/TinyLlama-1.1B-Chat-v0.1',
        'llama2-7b':  'meta-llama/Llama-2-7b-chat-hf',
        'llama3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    }
    if llama_version not in llm_model_map:
        raise ValueError(f"Unknown llama_version: {llama_version}. "
                         f"Choose from {list(llm_model_map.keys())}")
    llm = LLM(model_name=llm_model_map[llama_version])

    if freeze_llm:
        for param in llm.parameters():
            param.requires_grad = False

    if model_save_name == f'llm-{llama_version}':
        model = llm
    else:
        if llama_version == 'tiny_llama':
            model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
        else:
            model = GRetriever(llm=llm, gnn=gnn)

    print(f"Model device is: {llm.device}")

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': lr, 'weight_decay': 0.05}],
        betas=(0.9, 0.95)
    )

    grad_steps = 2
    best_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        if epoch == 0:
            print(f"Total Preparation Time: {time.time() - start_time:.2f}s")
            start_time = time.time()
            print("Training beginning...")

        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = get_loss(model, batch, model_save_name)
            loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch)
            optimizer.step()
            epoch_loss += float(loss)
            if (step + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']

        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = get_loss(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:.4f}")

        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best model...")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(
                model,
                f'{root_path}/models/{retrieval_config_version}_{algo_config_version}'
                f'_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt'
            )

    if llm.device.type != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{root_path}/models/{retrieval_config_version}_{algo_config_version}'
            f'_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt',
        )

    model.eval()
    eval_output = []
    print("Final evaluation...")
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            pred_time = time.time()
            pred = inference_step(model, batch, model_save_name)
            print(f"Time to predict: {time.time() - pred_time:.2f}s")
            eval_output.append({
                'pred': pred,
                'question': batch.question,
                'desc': batch.desc,
                'label': batch.label,
            })
        progress_bar_test.update(1)

    # Original metrics
    compute_metrics(eval_output)

    # Paper metrics: Hits@1, Hits@5, Recall@20, MRR
    compute_paper_metrics(eval_output)

    print(f"Total Training Time: {time.time() - start_time:.2f}s")
    save_params_dict(
        model,
        f'{root_path}/models/{retrieval_config_version}_{algo_config_version}'
        f'_{g_retriever_config_version}_{model_save_name}.pt'
    )
    torch.save(
        eval_output,
        f'{root_path}/models/{retrieval_config_version}_{algo_config_version}'
        f'_{g_retriever_config_version}_{model_save_name}_eval_outs.pt'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--llama_version', type=str, required=True,
                        choices=['tiny_llama', 'llama2-7b', 'llama3.1-8b'])
    parser.add_argument('--retrieval_config_version', type=int, required=True)
    parser.add_argument('--algo_config_version', type=int, required=True)
    parser.add_argument('--g_retriever_config_version', type=int, required=True)
    parser.add_argument('--encoder', type=str, default='gat',
                        choices=['gat', 'gcn'],
                        help='GNN encoder type (default: gat)')
    parser.add_argument('--freeze_llm', action='store_true',
                        help='Freeze LLM weights during training')
    parser.add_argument('--sys_prompt', type=str, default=None,
                        help='Optional system prompt override')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use')
    args = parser.parse_args()

    load_dotenv('db.env', override=True)
    start_time = time.time()

    train(
        args.epochs,
        args.gnn_hidden_channels,
        args.num_gnn_layers,
        args.batch_size,
        args.eval_batch_size,
        args.lr,
        llama_version=args.llama_version,
        retrieval_config_version=args.retrieval_config_version,
        algo_config_version=args.algo_config_version,
        g_retriever_config_version=args.g_retriever_config_version,
        encoder=args.encoder,
        checkpointing=args.checkpointing,
        sys_prompt=args.sys_prompt,
        num_gpus=args.num_gpus,
        freeze_llm=args.freeze_llm,
    )
    print(f"Total Time: {time.time() - start_time:.2f}s")
