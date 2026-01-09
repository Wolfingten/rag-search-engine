import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import load_data, CACHE_PATH

EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movies_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_EMBEDDINGS_META_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.documents = []
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Text is empty.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        contents = []
        for d in documents:
            self.document_map[d["id"]] = d
            contents.append(d["title"] + ": " + d["description"])
        self.embeddings = self.model.encode(contents, show_progress_bar=True)

        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for d in documents:
            self.document_map[d["id"]] = d

        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        if self.documents is None:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarities.append(
                (
                    cosine_similarity(query_embedding, embedding),
                    self.documents[i],
                )
            )
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        results = []
        for score, document in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        chunks = []
        chunks_meta = []
        for i, d in enumerate(documents):
            self.document_map[d["id"]] = d

            content = d.get("description", "")
            if not content:
                continue

            d_chunks = semantic_chunking(content, max_chunk_size=4, overlap=1)
            for j, c in enumerate(d_chunks):
                chunks.append(c)
                chunks_meta.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": j,
                        "total_chunks": len(d_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_meta

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_EMBEDDINGS_META_PATH, "w") as f:
            json.dump({"chunks": chunks_meta, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for d in documents:
            self.document_map[d["id"]] = d

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_EMBEDDINGS_META_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_EMBEDDINGS_META_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores = {}
        for c in chunk_scores:
            movie_idx = c["movie_idx"]
            if (movie_idx not in movie_scores) or (
                c["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = c["score"]

        movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for i, score in movie_scores[:limit]:
            d = self.documents[i]
            results.append(
                {
                    "id": d["id"],
                    "title": d["title"],
                    "document": d["description"][:100],
                    "score": score,
                }
            )
        return results


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunking(text: str, max_chunk_size: int = 100, overlap: int = 2):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2 and not sentences[0].endswith((".", "!", "?")):
        return sentences
    chunks = []
    i = 0
    while i + max_chunk_size < len(sentences):
        for s in sentences[i : i + max_chunk_size]:
            s = s.strip()
            if s:
                chunks.append(s)
        i = i + max_chunk_size - overlap
    chunks.append(sentences[i:])
    return chunks
