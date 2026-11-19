
from app.services.embedding_service import generate_embeddings
from app.services import vector_db
from app.utils.exceptions import RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def search_similar_chunks(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Search Qdrant for document chunks similar to the user's question.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    try:
        # Step 1: Convert the question into an embedding.
        query_embedding = generate_embeddings([query])[0]

        # Step 2: Make sure the collection exists.
        vector_db.create_collection()

        # Step 3: Connect to Qdrant.
        client = vector_db.get_qdrant_client()

        # Step 4: Search for similar vectors.
        response = client.query_points(
            collection_name=vector_db.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

        # Step 5: Format the search results.
        results = []

        for point in response.points:
            results.append({
                "chunk_id": str(point.id),
                "score": point.score,
                "document_id": point.payload.get("document_id"),
                "file_name": point.payload.get("file_name"),
                "page_number": point.payload.get("page_number"),
                "chunk_index": point.payload.get("chunk_index"),
                "text": point.payload.get("text")
            })

        logger.info(
            "Found %s relevant chunks for the search query",
            len(results)
        )

        return results

    except (ValueError, RetrievalError):
        raise

    except Exception as error:
        logger.exception("Semantic search failed")
        raise RetrievalError(
            "Failed to search document chunks."
        ) from error
