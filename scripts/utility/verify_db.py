from sqlalchemy import create_engine, text
import sys

def verify_db():
    from oml.config import DEFAULT_SQLITE_PATH
    db_path = f"sqlite:///{DEFAULT_SQLITE_PATH}"
    engine = create_engine(db_path)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT doc_id, summary FROM documents LIMIT 1"))
        row = result.fetchone()
        
        if row:
            print(f"Doc ID: {row[0]}")
            print(f"Summary: {row[1]}")
            if row[1]:
                print("SUCCESS: Summary found.")
            else:
                print("FAILURE: Summary is empty.")
                sys.exit(1)
        else:
            print("FAILURE: No documents found.")
            sys.exit(1)

if __name__ == "__main__":
    verify_db()
