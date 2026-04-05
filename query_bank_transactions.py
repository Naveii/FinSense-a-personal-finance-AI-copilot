from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def search_transactions(
    query: str,
    persist_directory: Path,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
) -> dict[str, Any]:
    embedding_model = SentenceTransformer(embedding_model_name)
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0].tolist()

    chroma_client = chromadb.PersistentClient(path=str(persist_directory))
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, transaction_id in enumerate(ids):
        matches.append(
            {
                "transaction_id": transaction_id,
                "distance": distances[index] if index < len(distances) else None,
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "document": documents[index] if index < len(documents) else "",
            }
        )

    return {
        "query": query,
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "embedding_model": embedding_model_name,
        "top_k": top_k,
        "matches": matches,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search bank transactions in ChromaDB using a natural-language query."
    )
    parser.add_argument("query", help='Natural-language query such as "show all large UPI debits".')
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DATA_DIR / "chroma_bank_transactions",
        help="Directory where the ChromaDB files are stored.",
    )
    parser.add_argument(
        "--collection-name",
        default="bank_transactions",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name or local path to use for query embeddings.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matching transactions to return.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    response = search_transactions(
        query=args.query,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
    )
    print(json.dumps(response, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
