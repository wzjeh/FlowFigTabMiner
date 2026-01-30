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
        # Construct Rows
        for point in raw_data:
            # Step 4 raw_data already has "Series" (name), "X", "Y_Left", "Y_Right/Data_Value"
            
            # Resolve Series Name
            raw_series = point.get("Series", "Unknown")
            # If Series name contains abbr, e.g. "PhCl", resolve it
            # Simple text resolution attempt
            resolved_series = resolve(raw_series)
            
            row = {
                "Series": resolved_series
            }
            
            # Add Axes
            # If we resolved X/Y titles, use them as keys? 
            # Or stick to standard "X_Value", "Y_Value" + "X_Unit/Desc" in separate metadata?
            # User typically wants the CSV to have the actual header as the column name.
            
            row[x_title] = point.get("X")
            
            # Handle Y values (heatmap might have Data Value)
            if "Y_Right/Data_Value" in point and point["Y_Right/Data_Value"] is not None:
                 # It's a heatmap/3D plot
                 row[y_title] = point.get("Y_Left") # The Y-axis position
                 row["Data_Value"] = point.get("Y_Right/Data_Value") # The Z-value (Color/Heat)
            else:
                 row[y_title] = point.get("Y_Left")
            
            # Add Global Conditions
            for k, v in conditions.items():
                row[f"Condition: {k}"] = v
            
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save CSV
        fid = meta.get("figure_id", "unknown")
        csv_path = os.path.join(self.output_dir, f"{fid}_rich.csv")
        df.to_csv(csv_path, index=False)
        print(f"[DataFusion] Saved rich CSV to {csv_path}")
        return csv_path
