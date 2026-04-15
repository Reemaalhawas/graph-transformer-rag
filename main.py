import argparse
import time
from dotenv import load_dotenv

from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GraphRAG GNN Architecture Comparison — STaRK-Prime'
    )
    parser.add_argument('--gnn_type', type=str, default='gat',
                        choices=['gat', 'gcn', 'graph_transformer'],
                        help='GNN architecture: gat (original), gcn (baseline), '
                             'or graph_transformer (most expressive)')
    parser.add_argument('--gnn_hidden_channels', type=int, default=1024)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--freeze_llm', action='store_true')
    parser.add_argument('--llama_version', type=str, default='llama3.1-8b',
                        choices=['tiny_llama', 'llama2-7b', 'llama3.1-8b'])
    parser.add_argument('--retrieval_config_version', type=int, required=True)
    parser.add_argument('--algo_config_version', type=int, required=True)
    parser.add_argument('--g_retriever_config_version', type=int, required=True)
    parser.add_argument('--max_txt_len', type=int, default=4096,
                        help='Max token length for textualized graph (4096 for A100 40GB)')
    args = parser.parse_args()

    load_dotenv('db.env', override=True)

    start_time = time.time()
    train(
        num_epochs=args.num_epochs,
        hidden_channels=args.gnn_hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        llama_version=args.llama_version,
        retrieval_config_version=args.retrieval_config_version,
        algo_config_version=args.algo_config_version,
        g_retriever_config_version=args.g_retriever_config_version,
        gnn_type=args.gnn_type,
        checkpointing=args.checkpointing,
        freeze_llm=args.freeze_llm,
        max_txt_len=args.max_txt_len,
    )
    print(f"Total Time: {time.time() - start_time:.2f}s")
