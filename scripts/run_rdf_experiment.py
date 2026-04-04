import os
import json
from oml.eval.fact_checker import SemanticFactChecker
from pathlib import Path

LOG_DIR = Path("logs/rdf")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run():
    print("Setting up RDF Semantic Fact Checking Experiment...")
    
    # 1. Define Ground Truth Triples (These would normally be extracted from the trusted corpus)
    ground_truth = [
        ("Victor Frankenstein", "created", "The Monster"),
        ("The Monster", "killed", "William Frankenstein"),
        ("Justine Moritz", "was executed for", "Murder"),
        ("Robert Walton", "captained", "The Ship"),
        ("Elizabeth Lavenza", "was murdered by", "The Monster")
    ]
    
    checker = SemanticFactChecker(use_llm=True)
    checker.add_facts(ground_truth)
    print(f"Loaded {len(ground_truth)} ground-truth facts into the RDF Graph.")
    
    # 2. Define LLM Responses (One accurate, one hallucinated)
    responses = [
        {
            "name": "Accurate Agent Response",
            "text": "Victor Frankenstein created The Monster in his lab. Later, The Monster killed William Frankenstein, which was tragic."
        },
        {
            "name": "Hallucinating Agent Response",
            "text": "Robert Walton created The Monster. Also, Justine Moritz was executed for treason, while Elizabeth Lavenza captained The Ship."
        }
    ]
    from oml.config import DEFAULT_MODEL
    model_name = os.getenv("OML_MODEL", DEFAULT_MODEL)
    
    log_path = LOG_DIR / "rdf_evaluation.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"--- SEMANTIC RDF FACT CHECKING EVALUATION (Model: {model_name}) ---\n\n")
        
        for idx, resp in enumerate(responses, 1):
            print(f"Evaluating Response {idx}: {resp['name']}")
            f.write(f"=== {resp['name']} ===\n")
            f.write(f"Generated Text: {resp['text']}\n\n")
            
            result = checker.verify(resp['text'], model_name)
            
            if result.get('status') == "success":
                f.write(f"Fact-Check Score: {result.get('score', 0):.2f}\n")
                f.write(f"Total Claims Extracted: {result.get('total_claims', 0)}\n")
                
                f.write("\n[PASS] Verified Claims (Supported by Graph):\n")
                for c in result.get('verified_claims', []):
                    f.write(f"  - {c}\n")
                    
                f.write("\n[FAIL] Unverified Claims / Hallucinations:\n")
                for c in result.get('unverified_claims', []):
                    f.write(f"  - CLAIM: {c['claim']}\n    REASON: {c['reason']}\n")
            else:
                f.write(f"Verification Failed: {result}\n")
                
            f.write("\n----------------------------------------\n\n")
            
    print(f"\nRDF Semantic Verification Complete.\nLogs written to: {log_path}")

if __name__ == "__main__":
    run()
