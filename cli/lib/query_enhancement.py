import os
import json
import time

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

from .utils import DEFAULT_SEARCH_LIMIT


def setup_gemini() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client


def prompt_gemini(prompt: str) -> str:
    client = setup_gemini()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return str(response.text)


def correct_spelling(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.
    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.

    Corrected:"""

    return prompt_gemini(prompt)


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""

    return prompt_gemini(prompt)


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """

    return prompt_gemini(prompt)


def rate_individual(query: str, document: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {document.get("title", "")} - {document.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""

    response = prompt_gemini(prompt)

    return 0.0 if response is None else float(response)


def rate_batch(query: str, documents: list) -> list:
    prompt = f"""Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {documents}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

    [75, 12, 34, 2, 1]

    Your response should start with '[' and end with ']'. No ther information besides the JSON list should be sent.
    """

    response = prompt_gemini(prompt)
    if response is None:
        return []

    response = response[response.find("[") : response.find("]") + 1]

    return json.loads(response)


def enhance_query(query: str, method: str = "") -> str:
    if method == "spell":
        enhanced_query = correct_spelling(query)
        print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    if method == "rewrite":
        enhanced_query = rewrite_query(query)
        print(f"Enhanced query (rewrite): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    if method == "expand":
        enhanced_query = expand_query(query)
        print(f"Enhanced query (expand): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    return query


def rerank_results(
    query: str,
    results: list[dict],
    method: str = "",
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict]:
    if method == "individual":
        for r in results:
            r["rerank_score"] = rate_individual(query, r)
            time.sleep(3)

        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:limit]

        for i, r in enumerate(results, 1):
            print(
                f"{i}. {r["title"]}\n\tRerank Score: {r["rerank_score"]}\n\tRRF Score: {r["rrf_score"]}\n\tBM25: {r["bm25_rank"]}, Semantic: {r["semantic_rank"]}\n\t{r["document"][:100]}"
            )
        return results

    if method == "batch":
        batch_scores = rate_batch(query, results)

        for r, s in zip(results, batch_scores):
            r["batch_score"] = s

        results = sorted(results, key=lambda x: x["batch_score"])[
            :limit
        ]  # Here we use ranks so we want to sort in increasing order

        for i, r in enumerate(results, 1):
            print(
                f"{i}. {r["title"]}\n\tRerank Score: {r["batch_score"]}\n\tRRF Score: {r["rrf_score"]}\n\tBM25: {r["bm25_rank"]}, Semantic: {r["semantic_rank"]}\n\t{r["document"][:100]}"
            )
        return results

    if method == "cross_encoder":
        pairs = [
            [
                query,
                f"{document.get('title', '')} - {document.get('document', '')}",
            ]
            for document in results
        ]

        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)

        for r, s in zip(results, scores):
            r["cross_score"] = s

        results = sorted(results, key=lambda x: x["cross_score"], reverse=True)[:limit]

        for i, r in enumerate(results, 1):
            print(
                f"{i}. {r["title"]}\n\tCross Encoder Score: {r["cross_score"]}\n\tRRF Score: {r["rrf_score"]}\n\tBM25: {r["bm25_rank"]}, Semantic: {r["semantic_rank"]}\n\t{r["document"][:100]}"
            )
        return results

    for i, r in enumerate(results, 1):
        print(
            f"{i}. {r["title"]}\n\tRRF Score: {r["rrf_score"]}\n\tBM25: {r["bm25_rank"]}, Semantic: {r["semantic_rank"]}\n\t{r["document"][:100]}"
        )
    return results
