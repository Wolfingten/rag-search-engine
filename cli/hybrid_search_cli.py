import argparse

from lib.utils import load_data
from lib.hybrid_search import min_max_normalize, HybridSearch
from lib.query_enhancement import (
    enhance_query,
    rerank_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a given array of values using Min/Max normalization.",
    )
    normalize_parser.add_argument(
        "array", nargs="+", type=float, help="Array to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Search movies based weighted and combined scores from keyword and semantic search.",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter to control weighting towards keyword or semantic search",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results returned"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Use hybrid search with reciprocal rank fusion of keyword and semantic search results.",
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="Parameter to control influence of high and low ranked results.",
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results returned"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Method to rerank most relevant search results.",
    )

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            rrf_search_command(
                args.query, args.enhance, args.rerank_method, args.k, args.limit
            )
        case "weighted-search":
            documents = load_data()
            index = HybridSearch(documents["movies"])
            results = index.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results, 1):
                r = r[1]
                print(
                    f"{i}. {r["title"]}\n\tHybrid Score: {r["hybrid_score"]}\n\tBM25: {r["bm25_score"]}, Semantic: {r["semantic_score"]}\n\t{r["document"][:100]}"
                )
        case "normalize":
            print(min_max_normalize(args.array))
        case _:
            parser.print_help()


def rrf_search_command(
    query: str, enhance: str = "", rerank_method: str = "", k=60, limit=5
):
    query = enhance_query(query, method=enhance)
    documents = load_data()
    index = HybridSearch(documents["movies"])

    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    if rerank_method:
        results = index.rrf_search(query, k, limit * 5)
        print(f"Reranking top {limit} results using {rerank_method} method...")

        results = rerank_results(query, results, method=rerank_method, limit=limit)

    results = index.rrf_search(query, k, limit)
    return results


if __name__ == "__main__":
    main()
