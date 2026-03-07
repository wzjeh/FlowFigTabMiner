import json
import re

class ContextResolver:
    """
    Step 5b: Context Resolution
    Resolves 'Unknown Terms' by searching definitions in the full PDF text using LLMs.
    """
    def __init__(self, full_text, external_context=None, llm_engine=None):
        self.full_text = full_text
        self.external_context = external_context or {}
        
        if llm_engine:
            self.llm = llm_engine
        else:
            from src.adjudication.llm_engine import LLMEngine
            self.llm = LLMEngine()

    def resolve(self, unknown_terms):
        """
        Resolve a list of unknown terms.
        """
        resolved = {}
        
        # 1. Check External Context
        remaining_terms = []
        for term in unknown_terms:
            found = False
            for k, v in self.external_context.items():
                if k.lower() == term.lower() or term.lower() in k.lower():
                     resolved[term] = v
                     found = True
                     break
            if not found and term in self.external_context:
                resolved[term] = self.external_context[term]
                found = True
            
            if not found:
                remaining_terms.append(term)

        # 2. Use LLM to resolve remaining terms against full text
        if not remaining_terms or not self.full_text:
            for t in remaining_terms:
                resolved[t] = "[Not Found]"
            return resolved
            
        print(f"      [ContextResolver] Resolving {len(remaining_terms)} terms via LLM against full text...")
        
        # Limit text size to prevent token explosion (usually definitions are in first half)
        text_chunk = self.full_text[:40000]
        
        sys_prompt = '''You are a chemical literature expert. Find the definition or full chemical name for these abbreviations in the text.
        Return ONLY valid JSON format mapping abbreviations to their full names. Do NOT wrap in markdown.
        If a term is fundamentally a standard unit or element (like "mL", "h", "H2") skip it or return its common meaning.
        If a term is not in the text, map it to "[Not Found]".
        Example: {"NB": "nitrobenzene", "1a": "[Not Found]"}'''
        
        usr_prompt = f"Terms to find: {remaining_terms}\n\nText:\n{text_chunk}"
        
        try:
             res = self.llm.chat(sys_prompt, usr_prompt)
             cleaned = res.strip()
             if "```" in cleaned:
                 cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
             if cleaned.startswith('json'):
                 cleaned = cleaned[4:].strip()
                 
             llm_resolved = json.loads(cleaned)
             for t in remaining_terms:
                 resolved[t] = llm_resolved.get(t) or llm_resolved.get(t.upper()) or llm_resolved.get(t.lower()) or "[Not Found]"
        except Exception as e:
             print(f"      [ContextResolver] LLM fallback failed: {e}")
             for t in remaining_terms:
                 resolved[t] = "[Not Found]"
                 
        return resolved
