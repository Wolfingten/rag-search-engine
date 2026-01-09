import argparse

from lib.semantic_search import ChunkedSemanticSearch, SemanticSearch, semantic_chunking
from lib.utils import load_data


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for a single text."
    )
    embed_parser.add_argument(
        "text", type=str, help="Text to generate an embedding for."
    )

    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset."
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query."
    )
    embed_query_parser.add_argument(
        "query", type=str, help="Query to search for in vector embeddings."
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies with semantic search."
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to return"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into chunks with optional overlap."
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Size of each chunk in number of words",
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Create chunks from sentence boundaries."
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Max size of each chunk in number of words",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap between chunks"
    )

    subparsers.add_parser("embed_chunks", help="Generate embeddings for chunked texts.")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunked embeddings."
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Return search limit"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = fixed_size_chunking(args.text, args.chunk_size)
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {" ".join(c)}")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            chunks = semantic_chunking(args.text, args.max_chunk_size, args.overlap)
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {" ".join(c)}")
        case "embed_chunks":
            embeddings = embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            results = search_chunked_embeddings(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r["title"]} (score: {r["score"]:.4f})")
                print(f"   {r["document"]}...")
        case _:
            parser.print_help()


def verify_model():
    index = SemanticSearch()
    print(f"Model loaded: {index.model}")
    print(f"Max sequence length: {index.model.max_seq_length}")


def embed_text(text):
    index = SemanticSearch()
    embedding = index.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    index = SemanticSearch()
    data = load_data()
    data = data["movies"]
    embeddings = index.load_or_create_embeddings(data)
    print(f"Number of docs:   {len(data)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    index = SemanticSearch()
    embedding = index.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def semantic_search(query, limit=10):
    index = SemanticSearch()
    data = load_data()
    data = data["movies"]
    index.load_or_create_embeddings(data)

    results = index.search(query, limit)

    for i, r in enumerate(results):
        print(f"{i}. {r[1]["title"]} (score: {r[0]})\n\t{r[1]["description"]}")


def fixed_size_chunking(text: str, chunk_size: int = 10, overlap: int = 2):
    words = text.split()
    chunks = []
    i = 0
    while i + chunk_size < len(words):
        chunks.append(words[i : i + chunk_size])
        i = i + chunk_size - overlap
    chunks.append(words[i:])
    return chunks


def embed_chunks():
    index = ChunkedSemanticSearch()
    data = load_data()
    data = data["movies"]
    return index.load_or_create_chunk_embeddings(data)


def search_chunked_embeddings(query: str, limit: int = 10):
    index = ChunkedSemanticSearch()
    data = load_data()
    data = data["movies"]
    index.load_or_create_chunk_embeddings(data)
    results = index.search_chunks(query, limit)
    return results


if __name__ == "__main__":
    main()
