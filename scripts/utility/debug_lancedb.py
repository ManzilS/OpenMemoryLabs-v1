import lancedb
import pandas as pd

try:
    print("Attempting to create LanceDB connection...")
    db = lancedb.connect("data/lancedb_debug")
    print("Success! Creating table...")
    
    data = [{"vector": [1.1, 1.2], "text": "foo", "id": "1"}]
    tbl = db.create_table("test", data, mode="overwrite")
    print("Table created.")
    
    res = tbl.search([1.1, 1.2]).limit(1).to_pandas()
    print("Search result:", res)
    
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
