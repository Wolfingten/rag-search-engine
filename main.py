import argparse

from config import SEARCH_LIMIT
from rag.cli.keyword_search_cli import search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = search_command(args.query, limit=SEARCH_LIMIT)
            for i, m in enumerate(result, 1):
                print(f"{i}. Movie Title {m["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
