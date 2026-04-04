from oml.llm.factory import get_llm_client

def generate_hypothetical_document(query: str, model_name: str) -> str:
    """
    Generates a hypothetical document using the given LLM model answering the query.
    This hallucinated document is then used for semantic vector search to find actual
    documents that are semantically similar.
    """
    prompt = f"""Please write a short, factual paragraph that directly answers the following query. 
Write it in the style of an informative reference document. Do not use conversational filler 
like 'Here is a paragraph' or 'Sure'.

Query: {query}

Hypothetical Document:
"""
    try:
        model = get_llm_client(model_name)
        response = model.generate(prompt)
        return response.strip()
    except Exception as e:
        print(f"[HyDE] Failed to generate hypothetical document: {e}")
        # On failure, default back to the raw query
        return query
