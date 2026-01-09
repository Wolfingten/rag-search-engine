import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .utils import DEFAULT_SEARCH_LIMIT


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float = 0.5, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        results_bm25 = self._bm25_search(query, limit * 500)
        results_semantic = self.semantic_search.search_chunks(query, limit * 500)
        scores_bm25 = min_max_normalize([r["score"] for r in results_bm25])
        scores_semantic = min_max_normalize([r["score"] for r in results_semantic])
        for result, score in zip(results_bm25, scores_bm25):
            result["norm_score"] = score
        for result, score in zip(results_semantic, scores_semantic):
            result["norm_score"] = score

        combined_results = {}
        for result in results_bm25:
            idx = result["id"]
            if idx not in combined_results:
                combined_results[idx] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if result["norm_score"] > combined_results[idx]["bm25_score"]:
                combined_results[idx]["bm25_score"] = result["norm_score"]

        for result in results_semantic:
            idx = result["id"]
            if idx not in combined_results:
                combined_results[idx] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if result["norm_score"] > combined_results[idx]["semantic_score"]:
                combined_results[idx]["semantic_score"] = result["norm_score"]

        for idx, data in combined_results.items():
            score_hybrid = hybrid_score(
                data["bm25_score"], data["semantic_score"], alpha
            )
            data["hybrid_score"] = score_hybrid

        return sorted(
            combined_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = DEFAULT_SEARCH_LIMIT):
        results_bm25 = self._bm25_search(query, (limit * 500))
        results_semantic = self.semantic_search.search_chunks(query, (limit * 500))

        combined_results = {}
        for rank, result in enumerate(results_bm25, 1):
            idx = result["id"]
            if idx not in combined_results:
                combined_results[idx] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": rank,
                    "semantic_rank": 0,
                    "rrf_score": rrf_score(rank, k),
                }

        for rank, result in enumerate(results_semantic, 1):
            idx = result["id"]
            if idx not in combined_results:
                combined_results[idx] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": 0,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }
            elif combined_results[idx]["rrf_score"] > 0.0:
                combined_results[idx]["semantic_rank"] = rank
                combined_results[idx]["rrf_score"] += rrf_score(rank, k)

        return sorted(
            combined_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:limit]


def min_max_normalize(array):
    if not array:
        return []

    min_score = min(array)
    max_score = max(array)

    if min_score == max_score:
        return [1.0] * len(array)

    norm = max_score - min_score
    result = [(s - min_score) / norm for s in array]

    return result


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)
