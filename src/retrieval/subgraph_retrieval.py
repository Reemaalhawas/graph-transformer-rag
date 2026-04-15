"""
Subgraph retrieval for GraphRAG.

Pipeline:
  1. Vector search   — find candidate nodes via cosine similarity
  2. PCST pruning    — extract a connected subgraph using
                       Prize-Collecting Steiner Tree (pcst_fast)
  3. Return          — subgraph node ids + edges for LLM context

The PCST step is optional (falls back to top-K vector search when
pcst_fast is not installed).
"""

import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple, Dict, Optional

try:
    import pcst_fast
    _PCST_AVAILABLE = True
except ImportError:
    _PCST_AVAILABLE = False


class SubgraphRetriever:
    """
    Retrieves a relevant subgraph for a given query embedding.

    Args:
        node_emb       : Pre-computed node embeddings  [N, D]
        node_ids       : Mapping from row-index → original node id  [N]
        edge_index     : Full graph edge_index  [2, E]
        edge_types     : Edge type labels  [E]  (optional)
        vector_top_k   : Number of candidate nodes from vector search
        pcst_prize     : Node prize value in PCST
        pcst_num_clusters : Number of components in PCST solution
    """

    def __init__(
        self,
        node_emb: Tensor,
        node_ids: Tensor,
        edge_index: Tensor,
        edge_types: Optional[Tensor] = None,
        vector_top_k: int = 50,
        pcst_prize: float = 1.0,
        pcst_num_clusters: int = 1,
    ):
        self.node_emb = node_emb                      # [N, D]  (L2-normalised)
        self.node_ids = node_ids.tolist()             # list[int]
        self.edge_index = edge_index                  # [2, E]
        self.edge_types = edge_types
        self.vector_top_k = vector_top_k
        self.pcst_prize = pcst_prize
        self.pcst_num_clusters = pcst_num_clusters
        self.num_nodes = node_emb.size(0)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def vector_search(self, query_emb: Tensor) -> Tuple[List[int], Tensor]:
        """
        Return the top-K node indices (row positions) and their scores.

        Args:
            query_emb  [D]  — L2-normalised query embedding

        Returns:
            (top_k_indices, scores)
        """
        # Cosine similarity (both are normalised → just dot product)
        scores = self.node_emb @ query_emb            # [N]
        k = min(self.vector_top_k, self.num_nodes)
        top_scores, top_idx = torch.topk(scores, k=k)
        return top_idx.tolist(), top_scores

    # ------------------------------------------------------------------
    # PCST subgraph extraction
    # ------------------------------------------------------------------

    def _pcst_subgraph(
        self,
        candidate_idx: List[int],
        candidate_scores: Tensor,
    ) -> List[int]:
        """
        Run PCST to find a connected subgraph from the candidates.

        Prize  = cosine similarity score for seed nodes, 0 for others.
        Cost   = uniform edge cost of 1.
        """
        if not _PCST_AVAILABLE:
            return candidate_idx[: self.vector_top_k]

        prizes = np.zeros(self.num_nodes, dtype=np.float64)
        for idx, score in zip(candidate_idx, candidate_scores.tolist()):
            prizes[idx] = float(score)

        # Build edge arrays expected by pcst_fast
        ei = self.edge_index.numpy()
        src = ei[0].astype(np.int64)
        dst = ei[1].astype(np.int64)
        num_edges = len(src)
        costs = np.ones(num_edges, dtype=np.float64)

        vertices, _ = pcst_fast.pcst_fast(
            src, dst, prizes, costs,
            self.pcst_num_clusters,
            "strong",
            0,
        )
        return vertices.tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_emb: Tensor,
        use_pcst: bool = True,
        max_nodes: int = 100,
    ) -> Dict:
        """
        Full retrieval pipeline for a single query.

        Args:
            query_emb  [D]      — L2-normalised query embedding
            use_pcst            — whether to apply PCST pruning
            max_nodes           — hard cap on returned nodes

        Returns:
            dict with keys:
              "node_idx"   : list[int]  — row indices in node_emb
              "node_ids"   : list[int]  — original knowledge-graph node ids
              "edge_index" : Tensor [2, E'] — subgraph edge_index
              "scores"     : dict[node_idx → score]
        """
        query_emb = query_emb.float()
        if query_emb.norm() > 0:
            query_emb = query_emb / query_emb.norm()

        cand_idx, cand_scores = self.vector_search(query_emb)

        if use_pcst and _PCST_AVAILABLE:
            node_idx = self._pcst_subgraph(cand_idx, cand_scores)
        else:
            node_idx = cand_idx

        # Cap size
        node_idx = node_idx[:max_nodes]
        node_set = set(node_idx)

        # Extract induced subgraph edges
        ei = self.edge_index
        mask = torch.tensor(
            [(s.item() in node_set and d.item() in node_set)
             for s, d in zip(ei[0], ei[1])],
            dtype=torch.bool,
        )
        sub_edge_index = ei[:, mask]

        # Score dict
        score_dict = {
            idx: cand_scores[i].item()
            for i, idx in enumerate(cand_idx)
            if idx in node_set
        }

        return {
            "node_idx":   node_idx,
            "node_ids":   [self.node_ids[i] for i in node_idx],
            "edge_index": sub_edge_index,
            "scores":     score_dict,
        }
