from app.services.embedding_service import generate_embeddings

texts = [
    "Artificial intelligence helps computers perform intelligent tasks.",
    "Machine learning allows computers to learn from data.",
    "Python is a programming language."
]

vectors = generate_embeddings(texts)

print("Number of vectors:", len(vectors))
print("Dimensions:", len(vectors[0]))
print("First vector sample:", vectors[0][:5])