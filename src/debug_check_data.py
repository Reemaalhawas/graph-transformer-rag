from src.loaders.prime_skb_loader import load_prime_skb
from src.loaders.prime_qa_loader import load_prime_qa
import ast


def main():
    # Load graph and QA
    skb = load_prime_skb()
    df, splits = load_prime_qa()

    node_info = skb["node_info"]

    print("Total questions:", len(df))
    print("Train/Val/Test sizes:",
          len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Look at one example question
    row = df.iloc[0]
    print("\n--- Example QA pair ---")
    print("Question ID :", row["id"])
    print("Question    :", row["query"])

    # STaRK uses column name 'answer_ids', not 'answer'
    # It's a string like "[15105, 15779, ...]"
    raw_answer_ids = row["answer_ids"]
    print("Raw answer_ids string:", raw_answer_ids)

    answer_ids = ast.literal_eval(raw_answer_ids)
    print("Parsed Answer IDs  :", answer_ids)

    print("\nAnswer node names:")
    for nid in answer_ids:
        info = node_info[nid]
        print(" ", nid, "->", info.get("name"), "| type:", info.get("type"))


if __name__ == "__main__":
    main()

