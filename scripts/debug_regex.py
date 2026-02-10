
import re

def debug_regex():
    with open('data/input/example_fulltext.txt', 'r') as f:
        full_text = f.read()

    term = '3,4-DCAN'
    safe_term = re.escape(term)
    print(f"Term: '{term}', Safe Term: '{safe_term}'")

    # The regex from context_resolver.py
    pattern_a = r"([a-zA-Z0-9\-\,\s]{3,50})\s*\(\s*" + safe_term + r"\s*\)"
    print(f"Pattern A: {pattern_a}")
    
    match = re.search(pattern_a, full_text, re.IGNORECASE)
    if match:
        print(f"Match A Found: '{match.group(0)}'")
        print(f"Captured Group 1: '{match.group(1)}'")
        candidate = match.group(1).strip()
        print(f"Split Candidate: {candidate.split()}")
        print(f"Len Candidate Split: {len(candidate.split())}")
    else:
        print("Match A Failed.")

    # Try simpler pattern
    pattern_simple = r"(.{1,50})\s*\(\s*3,4-DCAN\s*\)"
    match_s = re.search(pattern_simple, full_text, re.IGNORECASE)
    if match_s:
        print(f"Simple Match Found: '{match_s.group(0)}'")
    else:
        print("Simple Match Failed.")

    # Find position of term
    idx = full_text.find("(3,4-DCAN)")
    if idx != -1:
        print(f"Context Found at {idx}: '{full_text[max(0, idx-50):idx+20]}'")
    else:
        print("'(3,4-DCAN)' not found in text.")

if __name__ == "__main__":
    debug_regex()
