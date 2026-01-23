import os
import json
import glob
import re

class MetaAggregator:
    def __init__(self, evidence_dir):
        self.evidence_dir = evidence_dir

    def collect_terms(self):
        """
        Scans evidence JSONs and aggregates potential chemical terms/symbols.
        Returns:
            dict: {
                "terms": list of unique terms (candidates for resolution),
                "figures": list of figure_ids included
            }
        """
        json_pattern = os.path.join(self.evidence_dir, "*_evidence.json")
        evidence_files = glob.glob(json_pattern)
        
        all_terms = set()
        figure_ids = []
        
        print(f"[MetaAggregator] Found {len(evidence_files)} evidence files.")
        
        for json_file in evidence_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                meta = data.get("meta", {})
                fid = meta.get("figure_id", "unknown")
                figure_ids.append(fid)
                
                # 1. Harvest Text
                text_evidence = data.get("text_evidence", {})
                
                # Collect from all text fields
                for key in ['x_axis_title', 'y_axis_title', 'chart_text', 'legend_text']:
                    for item in text_evidence.get(key, []):
                        raw_text = item.get("text", "")
                        self._extract_from_text(raw_text, all_terms)
                        
                # Collect from caption
                caption = meta.get("caption", "")
                self._extract_from_text(caption, all_terms)
                
            except Exception as e:
                print(f"[MetaAggregator] Error reading {json_file}: {e}")
                
        # Clean and Filter Terms
        filtered_terms = self._filter_terms(all_terms)
        
        return {
            "terms": sorted(list(filtered_terms)),
            "figures": figure_ids
        }

    def _extract_from_text(self, text, term_set):
        """
        Splits text into potential tokens/symbols.
        """
        if not text: return
        
        # Heuristic: Split by logic, keep things that look like abbreviations.
        # e.g. "Yield of PhCl (%)" -> "Yield", "PhCl"
        
        # Remove parentheses content fully? No, "(%)" is noise but "(PhCl)" is gold.
        # Simple split by space first
        tokens = text.split()
        for t in tokens:
            # clean punctuation trailing/leading
            clean = t.strip(".,;:()[]{}")
            if len(clean) > 0:
                term_set.add(clean)

    def _filter_terms(self, term_set):
        """
        Remove common stopwords, numbers, etc.
        """
        stop_words = {
            "of", "the", "and", "in", "with", "for", "to", "by", "on", "at", 
            "figure", "fig", "table", "graph", 
            "yield", "selectivity", "conversion", "temperature", "pressure", # These are known standard terms, maybe keep them to confirm context?
            "time", "min", "h", "c", "k", "mpa", "bar" # Units
        }
        
        final_set = set()
        for t in term_set:
            t_lower = t.lower()
            if t_lower in stop_words: continue
            if t.replace('.', '').isdigit(): continue # Skip pure numbers
            if len(t) < 2 and not t.isupper(): continue # Skip single chars unless chemical symbol roughly (e.g. H, O, N... wait H is 1 char)
            
            final_set.add(t)
            
        return final_set
