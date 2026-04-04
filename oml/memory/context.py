from dataclasses import dataclass
from typing import List
import math

@dataclass
class ContextChunk:
    chunk_id: str
    text: str
    score: float

class ContextBudgeter:
    """
    Packs chunks into a prompt within a token budget and can report approximate token usage.
    """
    
    def __init__(self, chars_per_token: float = 3.5):
        # chars_per_token is a rough heuristic; adjust as needed for your models.
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Conservative token estimation."""
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)

    def construct_prompt(
        self, 
        query: str, 
        chunks: List[ContextChunk], 
        max_tokens: int = 4000, 
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer the user's question."
    ) -> str:
        """
        Constructs the final prompt string, prioritizing top chunks.
        """
        
        # 1. Structure Costs
        # We define the structure templates to measure their cost
        # Format:
        # <system>...</system>
        # <context>
        #   <chunk id="...">...</chunk>
        # </context>
        # <user>...</user>
        
        # Base overhead (wrapper tags)
        base_header = f"<system>\n{system_prompt}\n</system>\n\n<context>\n"
        base_footer = f"\n</context>\n\n<user>\n{query}\n</user>"
        
        fixed_cost = self.estimate_tokens(base_header) + self.estimate_tokens(base_footer)
        remaining_budget = max_tokens - fixed_cost
        
        if remaining_budget <= 0:
            # Emergency fallback: just user query
            return f"<user>\n{query}\n</user>"

        # 2. Pack Chunks
        packed_chunks_str = ""
        
        # Sort chunks by score descending (just in case they aren't)
        # Assuming input chunks are already sorted by relevance, but let's be safe if scores exist
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        
        for i, chunk in enumerate(sorted_chunks):
            # Use short ID (index+1) for token efficiency
            short_id = str(i + 1)
            
            # Chunk Wrapper: "  <chunk id='CID'>CONTENT</chunk>\n"
            # We estimate wrapper cost
            wrapper_overhead = self.estimate_tokens(f'  <chunk id="{short_id}"></chunk>\n')
            
            # Allow at least some content
            available_for_content = remaining_budget - wrapper_overhead
            
            if available_for_content < 10: 
                # Not worth adding a tiny slice or wrapper
                break
                
            chunk_tokens = self.estimate_tokens(chunk.text)
            
            if chunk_tokens <= available_for_content:
                # Add full chunk
                chunk_str = f'  <chunk id="{short_id}">{chunk.text}</chunk>\n'
                packed_chunks_str += chunk_str
                remaining_budget -= (chunk_tokens + wrapper_overhead)
            else:
                # Truncate
                # How many chars can we fit?
                allowed_chars = int(available_for_content * self.chars_per_token)
                truncated_text = chunk.text[:allowed_chars] + "...[truncated]"
                
                # Re-estimate to be safe (truncation adds chars)
                final_cost = self.estimate_tokens(f'  <chunk id="{short_id}">{truncated_text}</chunk>\n')
                
                if final_cost <= remaining_budget:
                     chunk_str = f'  <chunk id="{short_id}">{truncated_text}</chunk>\n'
                     packed_chunks_str += chunk_str
                     remaining_budget -= final_cost
                
                # Once we truncate/fill budget, we usually stop to avoid fragmenting context with low rank items
                break
                
        # 3. Assemble
        final_prompt = f"{base_header}{packed_chunks_str}{base_footer}"
        return final_prompt

    def construct_prompt_with_tokens(
        self,
        query: str,
        chunks: List[ContextChunk],
        max_tokens: int = 4000,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer the user's question.",
    ) -> tuple[str, int]:
        """Convenience helper: build prompt and return an approximate token count.

        The token count is estimated using the same heuristic as `estimate_tokens`.
        """
        prompt = self.construct_prompt(
            query=query,
            chunks=chunks,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        approx_tokens = self.estimate_tokens(prompt)
        return prompt, approx_tokens
