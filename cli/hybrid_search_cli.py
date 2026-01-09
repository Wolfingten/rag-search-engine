import argparse

from torch import norm

from lib.utils import load_data
from lib.hybrid_search import normalize, HybridSearch


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
        "--limit", type=int, help="Number of search results returned"
    )

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            documents = load_data()
            index = HybridSearch(documents["movies"])
            results = index.rrf_search(args.query, args.k, args.limit)
            for i, r in enumerate(results, 1):
                r = r[1]
                print(
                    f"{i}. {r["title"]}\n\tRRF Score: {r["rrf_score"]}\n\tBM25: {r["bm25_rank"]}, Semantic: {r["semantic_rank"]}\n\t{r["document"][:100]}"
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
            print(normalize(args.array))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
