from agentic.vectorstore.persistent_store import PersistentVectorStore

vector_store = PersistentVectorStore({})
collection = vector_store.collection

# Query for any documents (no filter)
results = collection.get(
    include=["metadatas", "documents"],
    limit=5
)

print(f"Found {len(results['ids'])} documents (no filter)")
for i, meta in enumerate(results["metadatas"]):
    print(f"\nDocument {i+1} metadata:")
    print(meta)
    print(f"Content: {results['documents'][i][:100]}...") 