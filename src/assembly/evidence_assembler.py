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

    def assemble(self, figure_id, extraction_data, intermediate_dir):
        """
        Assembles extraction results (Step 3) and OCR evidence (Step 2/4) into a JSON packet.
        
        Args:
            figure_id (str): Unique ID of the figure (e.g. 'page_3_figure_0_t0')
            extraction_data (list): List of dicts containing extracted data points (from CoordinateMapper)
            intermediate_dir (str): Path to directory containing crop images (macro_cleaned)
            
        Returns:
            str: Path to saved JSON, or None if filtered out.
        """
        print(f"[Assembler] Assembling evidence for {figure_id}...")
        
        # 1. Collect and OCR Text Crops
        text_evidence = self._process_text_crops(figure_id, intermediate_dir)
        
        # 2. Semantic Filtering (Step 4b Optimization)
        # "If image title or axis labels don't contain result information... directly discard"
        if not self._is_relevant_chart(text_evidence):
            print(f"      [Filter] Discarding {figure_id} due to lack of result keywords (yield, selectivity, etc).")
            return None

        # 3. Structure the Data
        # User Request: Aggregate chart_text into a single caption field
        caption_content = ". ".join([item['text'] for item in text_evidence.get('chart_text', [])])
        
        evidence_packet = {
            "meta": {
                "figure_id": figure_id,
                "source_intermediate_dir": intermediate_dir,
                "caption": caption_content # Added aggregated caption
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
        Keywords: yield, selectivity, conversion, etc.
        """
        import difflib
        
        # Target Keywords (Stricter list per user request)
        keywords = [
            'yield', 'uiel', 'yeild',           # yield
            'selectivity', 'slelctivity', 'selectiv', # selectivity
            'conversion', 'convers', 'conv.',   # conversion
            'recovery', 'purity', 'efficiency', # efficiency/purity
            'ee %', 'e.e.', 'dr',               # stereoselectivity
            # Removed: product, formation, ratio
        ]
        
        # Collect all text
        all_text_bits = []
        for key in ['x_axis_title', 'y_axis_title', 'chart_text', 'legend_text']:
            for item in text_evidence.get(key, []):
                all_text_bits.append(item['text'].lower())
        
        full_text = " ".join(all_text_bits)
        
        # 1. Direct Regex check for strong symbols
        # if '%' in full_text: return True # Too broad? "10%" might be in non-result charts.
        
        # 2. Token-based fuzzy match
        tokens = full_text.split()
        for token in tokens:
            # Clean token
            clean = re.sub(r'[^\w]', '', token)
            if len(clean) < 3: continue
            
            for kw in keywords:
                # Exact substring match (robust)
                if kw in clean:
                    return True
                
                # Fuzzy match for longer words
                if len(kw) > 4:
                    ratio = difflib.SequenceMatcher(None, clean, kw).ratio()
                    if ratio > 0.80:
                        print(f"      [Filter] Keyword Match: '{token}' ~= '{kw}' ({ratio:.2f})")
                        return True
                        
        # 3. Last resort: check if raw data exists? 
        # User said: "Image directly discarded". Even if we found data points, if axes are not relevant, we discard.
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
            elif "legend" in filename:
                key = "legend_text"
            else:
                continue # Skip raw, cleaned, etc.
            
            # Run OCR
            ocr_result = self.ocr.ocr(file_path)
            
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
            
            if full_text:
                # Basic cleanup (Step 4 optimization: logical filtering)
                if self._is_valid_evidence(full_text):
                    evidence[key].append({
                        "text": full_text,
                        "source_file": filename
                    })
        
        return evidence

    def _is_valid_evidence(self, text):
        """
        Filter out obviously useless OCR noise.
        """
        if len(text) < 2: return False # Skip single chars
        if text.replace('.', '').isdigit(): return False # Skip pure numbers (often axis ticks misdetected as titles)
        return True
