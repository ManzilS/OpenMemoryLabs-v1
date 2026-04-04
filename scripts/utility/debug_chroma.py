import chromadb
try:
    print("Attempting to create PersistentClient...")
    client = chromadb.PersistentClient(path="data/chroma_debug")
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
