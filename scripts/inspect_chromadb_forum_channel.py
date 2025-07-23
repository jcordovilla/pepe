from agentic.vectorstore.persistent_store import PersistentVectorStore

FORUM_CHANNEL_ID = '1384534612037865644'

vector_store = PersistentVectorStore({})
collection = vector_store.collection

# Query for documents with forum_channel_id
results = collection.get(
    where={"forum_channel_id": FORUM_CHANNEL_ID},
    include=["metadatas", "documents"],
    limit=5
)

print(f"Found {len(results['ids'])} documents with forum_channel_id = {FORUM_CHANNEL_ID}")
for i, meta in enumerate(results["metadatas"]):
    print(f"\nDocument {i+1} metadata:")
    print(meta)
    print(f"Content: {results['documents'][i][:100]}...") 