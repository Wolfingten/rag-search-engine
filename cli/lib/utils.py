import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

DEFAULT_SEARCH_LIMIT = 5


def load_data() -> dict:
    with open(
        DATA_PATH,
        "r",
        encoding="utf8",
    ) as f:
        data = json.load(f)
    return data
