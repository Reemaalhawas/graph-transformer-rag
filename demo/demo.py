# demo/demo.py
"""
Demo script for the GraphRAG pipeline.
Loads a trained checkpoint and runs inference on a biomedical question.

Usage:
  python demo/demo.py --gnn_type gat
  python demo/demo.py --gnn_type gat --question "What drugs treat strongyloidiasis?"
"""
import os, sys, argparse, time, torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
from neo4j import GraphDatabase
from models.gnn_variants import GCN, GraphTransformer

load_dotenv()
URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER", "neo4j")
PASS = os.getenv("NEO4J_PASSWORD")
FS   = "/home/ubuntu/GraphRAG"

DEMO_QUESTIONS = [
    "What drugs target the CYP3A4 enzyme and treat strongyloidiasis?",
    "Which genes interact with TP53 and are associated with breast cancer?",
    "What biological processes involve both insulin signaling and glucose metabolism?",
]


def load_model(gnn_type):
    from torch_geometric.nn import GAT
    ckpt = f"{FS}/qa_checkpoints/gnn-{gnn_type}-llm-llama3.1-8b/best_model.pt"
    assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}. Run training first."
    print(f"Loading {gnn_type} model from {ckpt}...")

    if gnn_type == "gat":
        gnn = GAT(in_channels=1536, hidden_channels=1024, num_layers=4, out_channels=1024)
    elif gnn_type == "gcn":
        gnn = GCN(in_channels=1536, hidden_channels=1024, num_layers=4, out_channels=1024)
    elif gnn_type == "graph_transformer":
        gnn = GraphTransformer(in_channels=1536, hidden_channels=1024,
                               num_layers=4, heads=4, out_channels=1024, edge_dim=1536)
    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}")

    try:
        from torch_geometric.nn import GRetriever
        from torch_geometric.nn.nlp import LLM
        llm   = LLM(model_name="meta-llama/Llama-3.1-8B-Instruct", num_params=8)
        model = GRetriever(llm=llm, gnn=gnn, use_lora=True)
    except ImportError:
        raise RuntimeError("GRetriever not available — check PyG version")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("Model loaded.")
    return model


def run_demo(model, question, driver):
    print(f"\nQuestion: {question}")
    print("-" * 65)
    t0 = time.time()

    from openai import OpenAI
    client = OpenAI()
    emb = client.embeddings.create(
        input=question, model="text-embedding-ada-002"
    ).data[0].embedding
    print(f"Step 1 — Query embedding: {time.time()-t0:.2f}s")

    t1 = time.time()
    r = driver.execute_query("""
        CALL db.index.vector.queryNodes('textembeddings', 4, $emb)
        YIELD node
        RETURN node.nodeId AS nodeId, node.name AS name
    """, parameters_={"emb": emb})
    seeds = [x.data() for x in r.records]
    print(f"Step 2 — Vector search ({time.time()-t1:.2f}s): {[s['name'] for s in seeds]}")

    names = [s['name'] for s in seeds]
    cypher = f"MATCH (n)-[r]->(m) WHERE n.name IN {names} RETURN n,r,m LIMIT 50"
    print(f"\nNeo4j Browser — paste to visualise the subgraph:")
    print(f"  {cypher}")
    print(f"\nTotal pipeline time: {time.time()-t0:.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn_type", default="gat",
                   choices=["gat", "gcn", "graph_transformer"])
    p.add_argument("--question", default=None)
    args = p.parse_args()

    driver = GraphDatabase.driver(URI, auth=(USER, PASS))
    model  = load_model(args.gnn_type)

    if args.question:
        run_demo(model, args.question, driver)
    else:
        print("\nAvailable demo questions:")
        for i, q in enumerate(DEMO_QUESTIONS, 1):
            print(f"  {i}. {q}")
        c = int(input("\nChoose (1-3): ")) - 1
        run_demo(model, DEMO_QUESTIONS[c], driver)

    driver.close()


if __name__ == "__main__":
    main()
