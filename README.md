# MoviePilot

MoviePilot is a **Retrieval-Augmented Generation (RAG)** search engine designed for movie datasets. It combines keyword-based retrieval with semantic, hybrid, and multimodal search techniques to provide relevant movie discovery and LLM-powered capabilities.

## Description

MoviePilot combines different traditional and modern information retrieval techniques from basic term matching to AI-driven retrieval:

* **Keyword Search:** Uses Okapi BM25 and TF-IDF for term matching.
* **Semantic Search:** Utilizes vector embeddings and chunking strategies (fixed and semantic) to understand the context of queries.
* **Hybrid Search:** Merges keyword and semantic results using **Reciprocal Rank Fusion (RRF)** and weighted scoring for balanced retrieval.
* **Multimodal Search:** Enables movie discovery using image-to-text and image-embedding capabilities.
* **RAG Pipeline:** Integrates with Google's Gemini LLM to generate summaries, answer questions, and provide cited responses based on retrieved movie documents.

## Motivation

The primary motivation for this project was to gain **hands-on experience** with the core components of modern search systems. It serves as a practical learning opportunity for:

* **Information Retrieval (IR):** Understanding the mathematical foundations of keyword search, such as term frequency (TF) and inverse document frequency (IDF).
* **RAG System Design:** Learning how to build an end-to-end pipeline that connects a retrieval engine to a Large Language Model (LLM) to reduce hallucinations and provide grounded answers.
* **Hybrid Strategies:** Experimenting with combining dense (semantic) and sparse (keyword) retrieval to overcome the limitations of each individual method.
* **Performance Tuning:** Learning how to evaluate search quality quantitatively using standard industry metrics.

## Quick Start

### 1. Prerequisites

Ensure you have Python 3.12+ installed.

### 2. Setup Environment

Clone the repository:

```bash
git clone https://github.com/jg-qbig/MoviePilot
cd MoviePilot

```

Create a `.env` file in the root directory. Add your Google GenAI API key:

```env
GEMINI_API_KEY=your_api_key_here

```

### 3. Install Dependencies

You can install the project and its dependencies using `pip`:

```bash
pip install .

```

### 4. Prepare the Data

The project expects a `movies.json` file in the `./data` directory.

1. Create the directory: `mkdir data`
2. Download the dataset: [movies.json](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json)
3. Place the downloaded file in `./data/movies.json`.

### 5. Initialize the Index

Before performing searches, build the initial keyword index:

```bash
python keyword_search_cli.py build

```

---
## Usage

MoviePilot provides several CLI entrypoints for different search and generation tasks located in `./cli`.

### 1. Keyword Search (`keyword_search_cli.py`)

Used for traditional information retrieval based on term frequency and document frequency.

| Command | Arguments | Use Case |
| --- | --- | --- |
| `build` | None | Builds and saves the inverted index database. |
| `bm25search` | `query`, `--limit` | Returns top matches using the Okapi BM25 algorithm. |
| `tfidf` | `doc_id`, `term` | Returns the TF-IDF score of a specific term in a document. |
| `idf` / `tf` | `term` / `doc_id`, `term` | Inspects raw term frequency or inverse document frequency. |

### 2. Semantic Search (`semantic_search_cli.py`)

Leverages vector embeddings to find movies based on meaning rather than just keywords.

| Command | Arguments | Use Case |
| --- | --- | --- |
| `search` | `query`, `--limit` | Standard semantic search using movie descriptions. |
| `semantic_chunk` | `text`, `--max-chunk-size` | Splits text into chunks based on sentence boundaries. |
| `search_chunked` | `query`, `--limit` | Searches movies using pre-processed chunked embeddings. |
| `embed_text` | `text` | Generates and displays the vector embedding for a string. |

### 3. Hybrid Search (`hybrid_search_cli.py`)

Combines keyword and semantic signals for superior accuracy.

| Command | Arguments | Use Case |
| --- | --- | --- |
| `weighted-search` | `query`, `--alpha` | Search with a manual weight between BM25 and Semantic scores. |
| `rrf-search` | `query`, `--k`, `--limit` | Hybrid search using Reciprocal Rank Fusion. |
| `rrf-search` | `--enhance [spell/rewrite]` | Performs search with query expansion or rewriting. |
| `rrf-search` | `--rerank-method` | Reranks top results using cross-encoders or batch LLM. |

### 4. Retrieval Augmented Generation (`augmented_generation_cli.py`)

The core RAG entrypoint that uses retrieved documents to generate natural language responses.

| Command | Arguments | Use Case |
| --- | --- | --- |
| `rag` | `query` | Performs hybrid search and generates a detailed response. |
| `summarize` | `query`, `--limit` | Generates an LLM-based summary of the top search results. |
| `citations` | `query`, `--limit` | Provides a summary with explicit citations to source movies. |
| `question` | `question`, `--limit` | Answers a specific question based on retrieved movie data. |

### 5. Multimodal Search & Evaluation

Tools for handling images and measuring performance.

| Script | Command / Flag | Use Case |
| --- | --- | --- |
| `multimodal_search_cli.py` | `image_search` | Finds movies similar to a provided image file. |
| `describe_image_cli.py` | `--image`, `--query` | Rewrites a query based on the content of an image. |
| `evaluation_cli.py` | `--limit` | Calculates Precision@k, Recall@k, and F1 scores against a golden dataset. |

---

## Contributing

Any pull requests or contributions are welcome.
