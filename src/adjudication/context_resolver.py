
import json
import re

class ContextResolver:
    """
    Step 5b: Context Resolution
    Resolves 'Unknown Terms' by searching definitions in the full PDF text.
    """
    def __init__(self, full_text, external_context=None):
        self.full_text = full_text
        self.external_context = external_context or {}

    def resolve(self, unknown_terms):
        """
        Resolve a list of unknown terms.
        Args:
            unknown_terms (list): ["NB", "3a", ...]
        Returns:
            dict: { "NB": "nitrobenzene", "3a": "product 3a (1,3-dinitrobenzene)", ... }
        """
        resolved = {}
        
        # 1. Check External Context (Agentic Brain)
        for term in unknown_terms:
            # Case-insensitive check
            for k, v in self.external_context.items():
                if k.lower() == term.lower() or term.lower() in k.lower(): # Loose match
                     resolved[term] = v
                     break
            # Exact match override
            if term in self.external_context:
                resolved[term] = self.external_context[term]

        # 2. Check Full Text (Regex Fallback)
        if not self.full_text:
            return resolved
            
        for term in unknown_terms:
            if term in resolved: continue # Already found
            
            definition = self._search_definition(term)
            if definition:
                resolved[term] = definition
            else:
                resolved[term] = "[Not Found]"
                
        return resolved

    def _search_definition(self, term):
        """
        Search for a term's definition in full text using regex patterns.
        """
        safe_term = re.escape(term)
        
        # Priority 1: "Full Name (Abbreviation)"
        # e.g. "3,4-dichloroaniline (3,4-DCA)"
        pattern_a = r"([a-zA-Z0-9\-\,\s]{3,50})\s*\(\s*" + safe_term + r"\s*\)"
        match = re.search(pattern_a, self.full_text, re.IGNORECASE)
        if match:
            # Check if the captured group is too long or contains invalid chars
            candidate_raw = match.group(1).strip()
            # If candidate spans multiple lines, take the last line (closest to term)
            if '\n' in candidate_raw:
                candidate_raw = candidate_raw.split('\n')[-1].strip()
            
            words = candidate_raw.split()
            # Heuristic: chemical names rarely > 5 words. If captured more, take last 5.
            if len(words) > 5:
                candidate = " ".join(words[-5:])
            else:
                candidate = candidate_raw
                
            # Filter out common junk like "the", "and", "product" if likely
            if candidate.lower() in ["the", "a", "an", "and", "or", "product", "compound"]:
                 return None
            
            return candidate
            
        # Priority 2: "Abbreviation (Full Name)"
        # e.g. "3,4-DCA (3,4-dichloroaniline)"
        pattern_b = safe_term + r"\s*\(\s*([a-zA-Z0-9\-\,\s]{3,50})\s*\)"
        match = re.search(pattern_b, self.full_text, re.IGNORECASE)
        if match:
             return match.group(1).strip()

        # Priority 3: "Full Name, Abbreviation" (often in lists)
        # e.g. "nitrobenzene, NB" - risky, might match random text.
        
        # Priority 4: "Abbreviation = Full Name" or "Abbreviation is Full Name"
        pattern_c = safe_term + r"\s*(?:=|is|refers to)\s*([a-zA-Z0-9\-\,\s]{3,50})"
        match = re.search(pattern_c, self.full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        return None
