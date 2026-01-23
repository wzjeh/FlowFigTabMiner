import os
import json
import pandas as pd
import re

class DataFusion:
    def __init__(self, output_dir="data/output"):
        self.output_dir = output_dir

    def fuse(self, evidence_path, knowledge_data):
        """
        Combines Step 4 Evidence with Step 5 Knowledge to produce Rich CSV.
        """
        with open(evidence_path, 'r') as f:
            evidence = json.load(f)
            
        raw_data = evidence.get("raw_data", [])
        meta = evidence.get("meta", {})
        text_ev = evidence.get("text_evidence", {})
        
        term_map = knowledge_data.get("term_mapping", {})
        conditions = knowledge_data.get("global_conditions", {})
        
        rows = []
        
        # Helper to resolve text
        def resolve(text):
            # Simple replacement of whole words?
            # Or check if text matches a key?
            if text in term_map:
                return f"{text} ({term_map[text]})"
            # Attempt substring replacement for strict keys?
            for abbr, full in term_map.items():
                if len(abbr) > 1 and f" {abbr} " in f" {text} ": # simplistic bound check
                     # Don't replace inside words
                     pass 
            return text

        # Prepare Legend Map (Series ID -> Text)
        legend_map = {}
        for item in text_ev.get("legend_text", []):
            fname = item["source_file"]
            # format: ..._legend_X.png
            match = re.search(r"_legend_(\d+)\.png", fname)
            if match:
                idx = int(match.group(1))
                txt = item["text"]
                # Resolve
                resolved_txt = resolve(txt)
                legend_map[idx] = resolved_txt
                
        # Prepare Axis Titles
        x_title = "X-Axis"
        if text_ev.get("x_axis_title"):
            x_title = resolve(text_ev["x_axis_title"][0]["text"])
            
        y_title = "Y-Axis"
        if text_ev.get("y_axis_title"):
            y_title = resolve(text_ev["y_axis_title"][0]["text"])

        # Construct Rows
        for point in raw_data:
            sid = point.get("series_id", -1) # Wait, coordinate mapper output format check needed
            # coordinate mapper: {"x": .., "y": .., "type": "data_point", "class": 0 (series)}
            # usually key is "class" or "series_id" depending on version.
            # Checking `coordinate_mapper.py` (not visible now, assuming "series_id" or "class")
            # Usually users scripts have "series_id".
            
            sid = point.get("series_id", point.get("class", -1))
            
            row = {
                x_title: point.get("x"),
                y_title: point.get("y"),
                "Series": legend_map.get(sid, f"Series {sid}")
            }
            
            # Add Global Conditions
            for k, v in conditions.items():
                row[f"Condition: {k}"] = v
            
            # Add Caption (optional, might be too long)
            # row["Caption"] = meta.get("caption", "")
            
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save CSV
        fid = meta.get("figure_id", "unknown")
        csv_path = os.path.join(self.output_dir, f"{fid}_rich.csv")
        df.to_csv(csv_path, index=False)
        print(f"[DataFusion] Saved rich CSV to {csv_path}")
        return csv_path
