from sqlalchemy import create_engine, text
import sys

def verify_chunks():
    from oml.config import DEFAULT_SQLITE_PATH
    db_path = f"sqlite:///{DEFAULT_SQLITE_PATH}"
    engine = create_engine(db_path)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT count(*) FROM chunks"))
        row = result.fetchone()
        
        count = row[0]
        print(f"Chunks found: {count}")
        
        if count > 0:
            print("SUCCESS: Chunks exist.")
        else:
            print("FAILURE: No chunks found.")
            sys.exit(1)

if __name__ == "__main__":
    verify_chunks()
