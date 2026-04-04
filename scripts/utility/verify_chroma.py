from oml.storage.chroma import ChromaStorage
from oml.models.schema import Document
import shutil
import os

def verify_chroma():
    persist_path = "data/chroma_test"
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)
        
    storage = ChromaStorage(persist_path=persist_path)
    
    doc = Document(
        doc_id="test_doc_1",
        raw_text="This is a test document for ChromaDB.",
        clean_text="This is a test document for ChromaDB.",
        summary="Test summary.",
        source="test_script"
    )
    
    print("Upserting document...")
    storage.upsert_documents([doc])
    
    print("Retrieving document...")
    retrieved = storage.get_document("test_doc_1")
    
    if retrieved:
        print(f"Retrieved Doc ID: {retrieved.doc_id}")
        print(f"Content: {retrieved.clean_text}")
        print(f"Summary: {retrieved.summary}")
        if retrieved.doc_id == "test_doc_1" and retrieved.summary == "Test summary.":
            print("SUCCESS: Document retrieved correctly.")
        else:
            print("FAILURE: Content mismatch.")
    else:
        print("FAILURE: Document not found.")

if __name__ == "__main__":
    verify_chroma()
