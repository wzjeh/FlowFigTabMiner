import os
import glob
import json
import re
from paddleocr import PaddleOCR

class EvidenceAssembler:
    def __init__(self, output_dir="data/evidence"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize PaddleOCR
        # use_angle_cls=True ensures we can read rotated y-axis text
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def check_relevance(self, figure_id, intermediate_dir):
        """
        Public method to check if a figure is relevant BEFORE running expensive Step 3.
        Returns: (is_relevant, text_evidence)
        """
        # 1. Collect and OCR Text Crops
        text_evidence = self._process_text_crops(figure_id, intermediate_dir)
        
        # 2. Semantic Filtering
        is_relevant = self._is_relevant_chart(text_evidence)
        
        if not is_relevant:
            # [RETRY] Try to enrich with Shared Captions
            print(f"      [Filter] Local check failed. Trying Context Expansion (Shared Caption)...")
            enriched = self._try_enrich_with_shared_caption(figure_id, intermediate_dir, text_evidence)
            
            if enriched:
                is_relevant = self._is_relevant_chart(text_evidence)
                if is_relevant:
                     print(f"      [Filter] SUCCESS! Saved by Shared Caption.")
                else:
                     print(f"      [Filter] FAILED even after Context Expansion.")
        
        return is_relevant, text_evidence

    def assemble(self, figure_id, extraction_data, intermediate_dir, text_evidence=None):
        """
        Assembles extraction results (Step 3) and OCR evidence (Step 2/4) into a JSON packet.
        
        Args:
            figure_id (str): Unique ID of the figure
            extraction_data (list): List of dicts containing extracted data points
            intermediate_dir (str): Path to directory containing crop images
            text_evidence (dict, optional): Pre-computed text evidence from check_relevance.
            
        Returns:
            str: Path to saved JSON, or None if filtered out.
        """
        print(f"[Assembler] Assembling evidence for {figure_id}...")
        
        # 1. Get Text Evidence (if not provided)
        if text_evidence is None:
            # Full check
            is_relevant, text_evidence = self.check_relevance(figure_id, intermediate_dir)
            if not is_relevant:
                 print(f"      [Filter] Discarding {figure_id} due to lack of result keywords.")
                 return None
        else:
            # Already checked, but let's just ensure we have it.
            # Assuming caller only calls assemble if relevant, 
            # OR we can re-verify if robust.
            pass

        # 3. Structure the Data
        # User Request: Aggregate chart_text into a single caption field
        caption_content = ". ".join([item['text'] for item in text_evidence.get('chart_text', [])])
        
        evidence_packet = {
            "meta": {
                "figure_id": figure_id,
                "source_intermediate_dir": intermediate_dir,
                "caption": caption_content
            },
            "text_evidence": text_evidence,
            "raw_data": extraction_data
        }
        
        # 4. Save JSON
        output_path = os.path.join(self.output_dir, f"{figure_id}_evidence.json")
        with open(output_path, 'w') as f:
            json.dump(evidence_packet, f, indent=2)
            
        return output_path

    def _is_relevant_chart(self, text_evidence):
        """
        Determines if the chart is relevant based on keywords in extracted text.
        Robust logic: 
        1. Normalizes all text (removes spaces/punctuation).
        2. Checks for presence of normalized keywords.
        """
        
        # Target Keywords (Normalized)
        # We look for these sequences inside the normalized text dump
        keywords = [
            'yield', 'uiel', 'yeild',           
            'selectivity', 'slelctivity', 'selectiv', 
            'conversion', 'convers', 
            'recovery', 'purity', 'efficiency', 
            'ee%', 'ee', 'dr',               
        ]
        
        # Collect all text
        all_text_bits = []
        for key in ['x_axis_title', 'y_axis_title', 'chart_text', 'legend_text']:
            for item in text_evidence.get(key, []):
                all_text_bits.append(item['text'].lower())
        
        full_text = "".join(all_text_bits) # Join without spaces first to preserve original order? 
        # Actually join with spaces then strip all non-alpha is safer for boundaries but we want to catch "S e l e c t i v i t y"
        # So: "S e l e c t i v i t y of Product" -> "selectivityofproduct"
        
        # Normalize: Remove all non-alphanumeric characters
        import re
        normalized_text = re.sub(r'[^a-z0-9]', '', full_text)
        
        print(f"      [Filter Debug] Normalized Text Dump (len={len(normalized_text)}): {normalized_text[:100]}...")
        
        for kw in keywords:
            # Check if keyword exists in normalized text
            if kw in normalized_text:
                print(f"      [Filter] Match Found: '{kw}'")
                return True
                
        # Fallback: Sliding window fuzzy match? 
        # For now, strict substring on normalized text handles the user's "Selectivity" (spaced) case perfectly.
        # "S e l e c t i v i t y" -> "selectivity" which contains "selectivity".
        
        print(f"      [Filter] FAILED. No keywords found in dump.")
        return False

    def _process_text_crops(self, figure_id, intermediate_dir):
        """
        Finds all text crops for the figure and runs OCR.
        Returns grouped text.
        """
        evidence = {
            "x_axis_title": [],
            "y_axis_title": [],
            "chart_text": [],
            "legend_text": []
        }
        
        # Define patterns to search specifically for this figure
        # Crops are named: {figure_id}_{label}_{index}.png
        # Note: figure_id itself might contain underscores, so we match by prefix
        
        search_pattern = os.path.join(intermediate_dir, f"{figure_id}_*.png")
        files = glob.glob(search_pattern)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # Determine type
            if "x_axis_title" in filename:
                key = "x_axis_title"
            elif "y_axis_title" in filename:
                key = "y_axis_title"
            elif "chart_text" in filename: # User mentioned "caption aka chart_text"
                key = "chart_text"
            # [FIX] Explicitly handle 'caption' files produced by YoloDetector
            elif "caption" in filename:
                key = "chart_text" # Treat caption as chart text for evidence
            elif "legend" in filename:
                key = "legend_text"
            else:
                continue # Skip raw, cleaned, etc.
            
            # Run OCR with Upscaling
            try:
                ocr_result = self._ocr_with_upscale(file_path)
            except Exception as e:
                print(f"      [OCR Error] {filename}: {e}")
                ocr_result = []
            
            # Parse OCR Result
            # Robust parsing (matching CoordinateMapper logic)
            txts = []
            if ocr_result and ocr_result[0]:
                # Case A: PaddleX Dict format
                if isinstance(ocr_result[0], dict) and 'rec_texts' in ocr_result[0]:
                    txts = ocr_result[0].get('rec_texts', [])
                # Case B: Standard PaddleOCR list format
                elif isinstance(ocr_result[0], list):
                    for line in ocr_result[0]:
                        if isinstance(line, list) and len(line) >= 2:
                            # line[1] is (text, conf)
                            txts.append(line[1][0])
            
            full_text = " ".join(txts).strip()
            print(f"      [OCR Debug] {filename} -> '{full_text}'")
            
            if full_text:
                # Basic cleanup
                if self._is_valid_evidence(full_text):
                    evidence[key].append({
                        "text": full_text,
                        "source_file": filename
                    })
        
        # Clean and Deduplicate
        self._clean_and_deduplicate(evidence)
        
        return evidence

    def _clean_and_deduplicate(self, evidence):
        """
        1. Remove exact/near duplicates within each category.
        2. Remove 'chart_text' that is actually a 'legend_text'.
        """
        # A. Deduplicate within category
        for key in evidence:
            unique_items = []
            seen_texts = set()
            
            for item in evidence[key]:
                # Normalize for comparison
                norm = re.sub(r'[^a-zA-Z0-9]', '', item['text'].lower())
                if not norm: continue
                
                if norm not in seen_texts:
                    seen_texts.add(norm)
                    unique_items.append(item)
                else:
                    # Duplicate found
                    # Optional: Keep the one with longer text or better conf?
                    # For now just keep first.
                    pass
            
            evidence[key] = unique_items
            
        # B. Cross-Category Cleaning
        # Remove chart_text if it matches any legend_text
        if evidence['legend_text'] and evidence['chart_text']:
            legend_norms = set()
            for l in evidence['legend_text']:
                legend_norms.add(re.sub(r'[^a-zA-Z0-9]', '', l['text'].lower()))
            
            cleaned_chart_text = []
            for c in evidence['chart_text']:
                c_norm = re.sub(r'[^a-zA-Z0-9]', '', c['text'].lower())
                
                # Check if this chart_text is basically a legend
                is_legend = False
                for l_norm in legend_norms:
                    if l_norm in c_norm or c_norm in l_norm:
                        # Heavy overlap?
                        # If ratio of overlap is high
                        if len(l_norm) > 5 and len(c_norm) > 5: # Ignore small noise
                             # SequenceMatcher ratio? or simple inclusion?
                             # Simple inclusion is risky for short words.
                             # But "3,4-dichloroaniline..." is long.
                             
                             # If almost identical
                             if l_norm == c_norm: 
                                 is_legend = True
                                 break
                             # If one contains the other and length diff is small
                             if abs(len(l_norm) - len(c_norm)) < 5:
                                 is_legend = True
                                 break
                
                if not is_legend:
                    cleaned_chart_text.append(c)
                else:
                    print(f"      [Cleaner] Removed chart_text that matched legend: '{c['text'][:20]}...'")
            
            evidence['chart_text'] = cleaned_chart_text

    def _try_enrich_with_shared_caption(self, figure_id, intermediate_dir, evidence):
        """
        Attempts to find a shared caption on the same page and add it to the evidence.
        Returns True if a new caption was found and added.
        """
        print(f"      [Fallback] Attempting to find shared caption for {figure_id}...")
        
        # extract page prefix e.g. "page_2" from "page_2_figure_2_t0"
        parts = figure_id.split('_')
        if len(parts) >= 2 and parts[0] == 'page':
            page_prefix = f"{parts[0]}_{parts[1]}" # "page_2"
            
            # Search for any caption file on this page: page_2_figure_*_caption_*.png
            # We want to find captions distinct from what we already have (if any)
            fallback_pattern = os.path.join(intermediate_dir, f"{page_prefix}_figure_*_caption_*.png")
            print(f"      [Fallback Debug] Scanning: {fallback_pattern}")
            fallback_files = sorted(glob.glob(fallback_pattern))
            print(f"      [Fallback Debug] Found {len(fallback_files)} candidates: {[os.path.basename(f) for f in fallback_files]}")
            
            # Check if we already have this file in our evidence
            current_files = set(item['source_file'] for item in evidence.get('chart_text', []))
            
            for f_path in fallback_files:
                 f_name = os.path.basename(f_path)
                 if f_name in current_files:
                     continue # Skip self
                 
                 # Found a new candidate!
                 print(f"      [Fallback] Borrowing shared caption: {f_name}")
                 try:
                    res = self._ocr_with_upscale(f_path)
                    txts = []
                    if res and res[0]:
                         if isinstance(res[0], dict) and 'rec_texts' in res[0]:
                             txts = res[0].get('rec_texts', [])
                         elif isinstance(res[0], list):
                             for line in res[0]:
                                 if isinstance(line, list) and len(line) >= 2:
                                     txts.append(line[1][0])
                    
                    fallback_text = " ".join(txts).strip()
                    print(f"      [Fallback Debug] OCR Result: '{fallback_text}'")
                    
                    if self._is_valid_evidence(fallback_text):
                         evidence["chart_text"].append({
                             "text": fallback_text,
                             "source_file": f_name,
                             "note": "Shared Caption Fallback"
                         })
                         return True # Enriched!
                 except Exception as e:
                     print(f"      [Fallback Error] {e}")
        
        return False

    def _ocr_with_upscale(self, file_path):
        """
        Reads image, upscales it 2x to improve OCR on small text, and runs PaddleOCR.
        """
        try:
            import cv2
            import numpy as np
            
            img = cv2.imread(file_path)
            if img is None:
                # Fallback to direct path processing if read fails
                return self.ocr.ocr(file_path)
            
            # Upscale 3x (Better for small fonts)
            h, w = img.shape[:2]
            scale = 3 
            # Use Cubic for better text definition
            img_scaled = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
            
            # Add Padding (White Border) - Crucial for text near edges!
            pad = 50
            img_padded = cv2.copyMakeBorder(
                img_scaled, pad, pad, pad, pad, 
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            
            return self.ocr.ocr(img_padded)
        except Exception as e:
            print(f"      [OCR Upscale Error] {e}. Fallback to raw.")
            return self.ocr.ocr(file_path)

    def _is_valid_evidence(self, text):
        """
        Filter out obviously useless OCR noise.
        """
        if len(text) < 2: return False # Skip single chars
        if text.replace('.', '').isdigit(): return False # Skip pure numbers (often axis ticks misdetected as titles)
        return True
