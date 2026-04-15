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
from torch_geometric.nn import GRetriever
from torch_geometric.nn.nlp import LLM
from tqdm import tqdm

from compute_metrics import compute_metrics

from STaRKQADataset import STaRKQADataset
from STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset

from torch_geometric.nn import GAT
from src.models import build_gcn, GraphTransformerEncoder, CombinedEncoder


def build_gnn(encoder: str, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, heads: int) -> torch.nn.Module:
    """
    Return the requested GNN encoder.
    All four are drop-in replacements inside GRetriever.
    """
    if encoder == 'gat':
        return GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
        )
    elif encoder == 'gcn':
        return build_gcn(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
    elif encoder == 'graph_transformer':
        return GraphTransformerEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
        )
    elif encoder == 'combined':
        return CombinedEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
        )
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: gat, gcn, graph_transformer, combined")


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

FS_CKPT = "/home/ubuntu/GraphRAG/checkpoints"

def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]
    # Save to local path
    torch.save(state_dict, save_path)
    # Also save to persistent filesystem
    os.makedirs(FS_CKPT, exist_ok=True)
    fs_path = os.path.join(FS_CKPT, os.path.basename(save_path))
    torch.save(state_dict, fs_path)
    print(f"Checkpoint also saved to filesystem: {fs_path}")

def load_params_dict(model, save_path):
    state_dict = model.state_dict()
    state_dict.update(torch.load(save_path, weights_only=True))
    model.load_state_dict(state_dict)
    return model


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
    encoder='combined',
    checkpointing=False,
    freeze_llm=False,
    num_gpus=None,
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

    print("Loading stark-qa prime train dataset...")
    t = time.time()

    if num_gnn_layers == 0:
        model_save_name = f'llm-{llama_version}'
    else:
        model_save_name = f'{encoder}-llm-{llama_version}'

    if model_save_name == f'llm-{llama_version}':
        root_path = f"stark_qa_vector_rag_{retrieval_config_version}"
        train_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_train, split="train")
        print(f'Finished loading train dataset in {time.time() - t:.1f}s')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_val, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_test, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)
    else:
        root_path = f"stark_qa_v{retrieval_config_version}_{algo_config_version}"
        train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version, algo_config_version, split="train")
        print(f'Finished loading train dataset in {time.time() - t:.1f}s')
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

    # ------------------------------------------------------------------ LLM
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

    # ------------------------------------------------------------------ GNN
    if num_gnn_layers > 0:
        gnn = build_gnn(
            encoder=encoder,
            in_channels=1536,          # Ada-002 node embeddings from Neo4j
            hidden_channels=hidden_channels,
            out_channels=1536,
            num_layers=num_gnn_layers,
            heads=4,
        )
        print(f"GNN encoder: {encoder}  |  layers: {num_gnn_layers}  |  hidden: {hidden_channels}")
        gnn_params = sum(p.numel() for p in gnn.parameters())
        print(f"GNN parameters: {gnn_params:,}")

    # ---------------------------------------------------------------- Model
    if model_save_name == f'llm-{llama_version}':
        model = llm
    else:
        if llama_version == 'tiny_llama':
            model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
        else:
            model = GRetriever(llm=llm, gnn=gnn)

    print(f"Model device: {llm.device}")

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': 0.05}],
                                  betas=(0.9, 0.95))
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

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = get_loss(model, batch, model_save_name)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:.4f}")

        if checkpointing and val_loss < best_val_loss:
            print("Checkpointing best model...")
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(
                model,
                f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt'
            )

    if llm.device.type != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    if checkpointing and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt',
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
            eval_data = {
                'pred': pred,
                'question': batch.question,
                'desc': batch.desc,
                'label': batch.label,
            }
            eval_output.append(eval_data)
        progress_bar_test.update(1)

    compute_metrics(eval_output)
    print(f"Total Training Time: {time.time() - start_time:.2f}s")
    save_params_dict(
        model,
        f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}.pt'
    )
    torch.save(
        eval_output,
        f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_eval_outs.pt'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='gat',
                        choices=['gat', 'gcn', 'graph_transformer', 'combined'],
                        help='GNN encoder to use (replaces GAT from original repo)')
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--freeze_llm', action='store_true',
                        help='Freeze LLM weights and train GNN only')
    parser.add_argument('--llama_version', type=str, default='tiny_llama',
                        choices=['tiny_llama', 'llama2-7b', 'llama3.1-8b'])
    parser.add_argument('--retrieval_config_version', type=int, default=0)
    parser.add_argument('--algo_config_version', type=int, default=0)
    parser.add_argument('--g_retriever_config_version', type=int, default=0)
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
        freeze_llm=args.freeze_llm,
    )
    print(f"Total Time: {time.time() - start_time:.2f}s")
