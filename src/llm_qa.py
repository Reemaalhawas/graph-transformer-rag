"""
LLM-based Q&A over retrieved subgraphs.

Given a natural-language query and a set of retrieved nodes/edges,
this module formats the subgraph as text context and calls an LLM
to generate the answer (GraphRAG-style).
"""

import os
from typing import List, Dict, Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are a biomedical knowledge graph expert.
You will be given:
  1. A question
  2. A relevant subgraph extracted from a biomedical knowledge graph

Use ONLY the information in the subgraph to answer the question.
If the answer is not in the subgraph, say "I don't know based on the provided context."
Keep your answer concise and factual."""


def format_subgraph_context(
    node_ids: List[int],
    node_info: Dict,
    edge_index,
    edge_type_dict: Dict,
    edge_types=None,
    max_nodes: int = 50,
    max_edges: int = 100,
) -> str:
    """
    Convert a subgraph into a readable text context for the LLM.
    """
    # Nodes section
    lines = ["### Nodes"]
    for i, nid in enumerate(node_ids[:max_nodes]):
        info = node_info.get(nid, {})
        name = info.get("name", f"node_{nid}")
        ntype = info.get("type", "unknown")
        lines.append(f"  - [{ntype}] {name} (id={nid})")

    # Edges section
    node_set = set(node_ids[:max_nodes])
    lines.append("\n### Relationships")
    edge_count = 0
    ei = edge_index
    for i in range(ei.size(1)):
        src, dst = ei[0, i].item(), ei[1, i].item()
        if src not in node_set or dst not in node_set:
            continue
        rel = "relates_to"
        if edge_types is not None and edge_type_dict:
            rel = edge_type_dict.get(edge_types[i].item(), "relates_to")
        src_name = node_info.get(src, {}).get("name", f"node_{src}")
        dst_name = node_info.get(dst, {}).get("name", f"node_{dst}")
        lines.append(f"  - {src_name} --[{rel}]--> {dst_name}")
        edge_count += 1
        if edge_count >= max_edges:
            break

    return "\n".join(lines)


class GraphRAGQA:
    """
    GraphRAG question-answering pipeline.

    Combines:
      1. GNN-based subgraph retrieval
      2. LLM-based answer generation over the retrieved context
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(
        self,
        question: str,
        subgraph_context: str,
    ) -> str:
        """
        Generate an answer given the question and subgraph context string.
        """
        user_msg = f"Question: {question}\n\nSubgraph context:\n{subgraph_context}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def run_pipeline(
        self,
        question: str,
        retrieval_result: Dict,
        node_info: Dict,
        edge_type_dict: Dict,
    ) -> Dict:
        """
        Full GraphRAG pipeline.

        Args:
            question         : Natural language question
            retrieval_result : Output from SubgraphRetriever.retrieve()
            node_info        : Full node metadata dict
            edge_type_dict   : Edge type id → name mapping

        Returns:
            dict with "answer", "node_ids", "context"
        """
        context = format_subgraph_context(
            node_ids=retrieval_result["node_ids"],
            node_info=node_info,
            edge_index=retrieval_result["edge_index"],
            edge_type_dict=edge_type_dict,
        )
        answer = self.answer(question, context)

        return {
            "answer":   answer,
            "node_ids": retrieval_result["node_ids"],
            "context":  context,
        }
