import math
import os
import pickle
from string import punctuation
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .utils import PROJECT_ROOT, CACHE_PATH, load_data

SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def search_command(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()
    seen, result = set(), []
    processed_query = tokenize(query)
    for q in processed_query:
        matches = index.get_documents(q)
        for i in matches:
            if i in seen:
                continue
            seen.add(i)
            result.append(index.docmap[i])
            if len(result) >= limit:
                return result
    return result


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r", encoding="utf8") as f:
        stopwords = f.read()
    return stopwords.splitlines()


def tokenize(text: str) -> list[str]:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    words = []
    for w in text.split():
        if w not in stopwords:
            w = stemmer.stem(w)
            words.append(w)
    return words


def matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for q in query_tokens:
        for t in title_tokens:
            if q in t:
                return True
    return False


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap: dict[int, dict] = {}
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths: dict[int, int] = {}
        self.doc_length_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for t in set(tokens):
            self.index[t].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if self.doc_lengths:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return 0.0

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return sorted(list(self.index.get(token, set())))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        n_total = len(self.docmap)
        n_matches = len(self.get_documents(token))
        return math.log((n_total + 1) / (n_matches + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25_idf(self, term: str) -> float:
        n_total = len(self.docmap)
        n_matches = len(self.get_documents(term))
        return math.log((n_total - n_matches + 0.5) / (n_matches + 0.5) + 1)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = SEARCH_LIMIT) -> list:
        tokens = tokenize(query)

        scores = {}
        for d_id in self.docmap:
            total_score = 0
            for t in tokens:
                total_score += self.bm25(d_id, t)
            scores[d_id] = total_score
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for i, score in scores[:limit]:
            d = self.docmap[i]
            results.append(
                {
                    "id": d["id"],
                    "title": d["title"],
                    "document": d["description"],
                    "score": score,
                }
            )
        return results

    def build(self):
        data = load_data()
        for movie in data["movies"]:
            content = movie["title"] + "\n" + movie["description"]
            self.__add_document(movie["id"], content)
            self.docmap[movie["id"]] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_length_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_length_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
