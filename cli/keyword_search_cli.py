import argparse

from utils import search_command, InvertedIndex, BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    tf_parser = subparsers.add_parser(
        "tf",
        help="Return the number of occurrences of a term {term} in a document {doc_id}",
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser(
        "idf",
        help="Return the inverse document frequency of a term {term}.",
    )
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Return the TF-IDF score of a term {term}.",
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Return the IDF score of a term {term} using Okapi BM25 algorithm.",
    )
    bm25idf_parser.add_argument("term", type=str, help="Search term")

    bm25tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Return the TF score of a term {term} using Okapi BM25 saturation.",
    )
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Search term")
    bm25tf_parser.add_argument(
        "--k1",
        type=float,
        default=BM25_K1,
        help="Parameter to adjust frequency saturation strength.",
    )
    bm25tf_parser.add_argument(
        "--b",
        type=float,
        default=BM25_B,
        help="Parameter to adjust document length normalization strength.",
    )

    subparsers.add_parser(
        "build", help="Build the inverted index database and save it to disk"
    )

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25idf":
            index = InvertedIndex()
            index.load()
            bm25idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tfidf = index.get_tfidf(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )
        case "idf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for res in results:
                print(f"{res["id"]}. {res["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
