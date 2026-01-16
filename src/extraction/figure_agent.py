import json
import os
import re
from src.llm_factory import LLMFactory

class FigureAgent:
    def __init__(self):
        # Keywords to identify relevant charts (case-insensitive)
        self.KEYWORDS = ["yield", "conversion", "selectivity", "ee%", "d.r.", "enantiomeric excess", "diastereomeric ratio"]
        
    def process(self, yolo_item):
        """
        Main entry point for processing a chart candidate from YOLO.
        
        Args:
            yolo_item (dict): Output from YoloDetector, containing:
                              - 'original_source': str
                              - 'cleaned_image': str (path to white-masked image)
                              - 'elements': dict of list of paths (keys: 'chart_text', 'legend', etc.)
        
        Returns:
            dict: Extraction result with keys: 'is_valid', 'reason', 'data', 'code', etc.
        """
        print(f"   [FigureAgent] Processing {os.path.basename(yolo_item['cleaned_image'])}...")
        
        # 1. Keyword Filtering
        # We look at 'chart_text' crops mostly. If none, maybe check whole image context? 
        # For now, rely on chart_text crops.
        if not self._check_relevance(yolo_item.get("elements", {})):
            print(f"      -> Filtered out (No relevant keywords found).")
            return {"is_valid": False, "reason": "No relevant keywords (Yield/Conversion/etc) found"}
        
        print(f"      -> Relevant chart detected.")

        # 2. Legend Analysis
        # We look at 'legend' crops to understand markers.
        legend_crops = yolo_item.get("elements", {}).get("legend", [])
        parsed_legends = self._analyze_legends(legend_crops)
        print(f"      -> Parsed {len(parsed_legends)} legend entries.")

        # 3. Code Generation & Extraction
        # We work on the 'cleaned_image' where legends/text are masked to avoid interference.
        extraction_result = self._generate_and_extract(yolo_item["cleaned_image"], parsed_legends)
        
        return {
            "is_valid": True,
            "relevant_keywords_found": True,
            "legends": parsed_legends,
            "extraction": extraction_result
        }

    def _check_relevance(self, elements_dict):
        """
        Returns True if keywords found in chart_text OR axis titles.
        If no text/axis images found, defaults to TRUE (permissive) with warning to allow testing.
        """
        # Gather all text-containing crops
        text_paths = elements_dict.get("chart_text", []) + \
                     elements_dict.get("x_axis_title", []) + \
                     elements_dict.get("y_axis_title", [])

        if not text_paths:
            print("      -> [Warning] No text/axis crops found. Assuming relevant for testing.")
            return True

        full_text = ""
        for p in text_paths:
            prompt = "Transcribe all text in this image precisely."
            try:
                # Using LLMFactory with image
                response = LLMFactory.create_completion(prompt, image_path=p)
                full_text += " " + response
            except Exception as e:
                print(f"      [Warning] OCR failed for {os.path.basename(p)}: {e}")
        
        print(f"      -> OCR Text: {full_text[:100]}...")
        full_text_lower = full_text.lower()
        
        for kw in self.KEYWORDS:
            if kw.lower() in full_text_lower:
                print(f"      -> Found keyword: '{kw}'")
                return True
        
        print("      -> No keywords found in OCR text.")
        # For strict filtering later, return False. 
        # But maybe 'Yield' is just 'Y' or implied? 
        # User said: "filter mechanism... if not found... don't use agent".
        # So I should return False if text IS found but keywords ARE NOT.
        return False

    def _analyze_legends(self, legend_paths):
        """
        Ask VLM to describe markers in legend crops.
        Returns list of dicts: [{'label': 'Yield', 'color': 'blue', 'shape': 'circle'}]
        """
        if not legend_paths:
            return []
            
        results = []
        for p in legend_paths:
            prompt = """
            Analyze this chart legend. 
            Identify every legend entry. For each, extract:
            1. The Label text (e.g., "Yield", "Product A").
            2. The Visual Marker description (Color and Shape).
            
            Return JSON list ONLY:
            [
              {"label": "...", "marker_visuals": "..."}
            ]
            """
            try:
                response = LLMFactory.create_completion(prompt, image_path=p)
                # Clean code blocks
                clean_json = self._clean_json_text(response)
                data = json.loads(clean_json)
                if isinstance(data, list):
                    results.extend(data)
            except Exception as e:
                print(f"      [Warning] Legend analysis failed for {os.path.basename(p)}: {e}")
        
        return results

    def _get_axis_ranges(self, image_path):
        """
        Ask VLM to identifying the numerical range of X and Y axes.
        Returns: {'x': [min, max], 'y': [min, max]} or None
        """
        prompt = """
        Analyze this chart image carefully.
        Identify the numerical range of the X-axis and Y-axis.
        Look at the numbers on the axes.
        
        Return JSON object ONLY:
        {
          "x_min": <number>,
          "x_max": <number>,
          "y_min": <number>,
          "y_max": <number>
        }
        If you cannot determine a value clearly, use null.
        """
        try:
            response = LLMFactory.create_completion(prompt, image_path=image_path)
            clean_json = self._clean_json_text(response)
            data = json.loads(clean_json)
            # basic validation
            return data
        except Exception as e:
            print(f"      [Warning] Axis range detection failed: {e}")
            return None

    def _generate_and_extract(self, image_path, legends_info):
        """
        1. Get Axis Ranges (VLM).
        2. Generate Code (LLM) with range context.
        3. Execute Code (Subprocess).
        """
        # Step 1: Get Ranges
        axis_ranges = self._get_axis_ranges(image_path)
        range_context = ""
        if axis_ranges:
            range_context = f"""
            **Observed Axis Ranges** (Use these for mapping):
            - X-axis: {axis_ranges.get('x_min')} to {axis_ranges.get('x_max')}
            - Y-axis: {axis_ranges.get('y_min')} to {axis_ranges.get('y_max')}
            """
        else:
            range_context = "Could not automatically detect axis ranges. Use reasonable placeholders or attempt to detect from text."

        legend_str = json.dumps(legends_info, indent=2)
        
        prompt = f"""
        You are an expert Python Data Scientist specializing in Computer Vision.
        
        TASK: Extract scatter plot data from the provided chart image.
        CONTEXT: 
        - The image has legends and titles MASKED (white boxes) to reduce noise.
        - You are provided with the KNOWN Legend Mapping:
        {legend_str}
        {range_context}
        
        CRITICAL INSTRUCTION:
        **Coordinate System**: Do NOT rely on OCR of axis numbers alone.
        You MUST detect the **TICK MARKS** (small tick lines on the axes) to establish the precise pixel-to-data mapping.
        
        **Mapping Logic**:
        1. Detect the pixel positions of the `min` and `max` ticks.
        2. Map them to the provided `Observed Axis Ranges`.
        3. **FALLBACK**: If you cannot find ticks, assume the plot area covers the central 80% of the image and map the ranges to that. **DO NOT CRASH** if ticks are missing.
        
        **Coding Guidelines (CRITICAL for Success)**:
        1. **Axis Detection**: Do NOT assume the axis is at the very edge (pixel 0 or height-1).
           - **Step A**: Find the **Main Axis Lines** (the longest horizontal line near the bottom and veritcal line near the left).
           - **Step B**: Detect **Ticks** (short perpendicular lines) strictly *along* those main lines.
        2. **Color Detection**:
           - Use **Wide Color Ranges** in HSV (e.g., Hue +/- 20, Saturation > 50).
           - If standard colors fail, try to detect *any* significant clusters of non-black/white points.
        3. **Robustness**:
           - Use `cv2.HoughLinesP` for better line detection.
           - If no ticks are found, fallback gracefully as previously instructed.
           - Ensure `try-except` blocks wrap every major CV step.

        OUTPUT:
        Write a complete, self-contained Python script using `cv2`, `numpy`, and `sklearn`.
        The script MUST print a valid JSON to stdout as its final action.
        
        Wrap the code in ```python ... ```.
        """
        
        try:
            # Step 2: Generate Code
            response = LLMFactory.create_completion(prompt, image_path=image_path)
            code = self._extract_code_block(response)
            
            # Step 3: Execute Code
            # We need to make sure the code uses the absolute path or correct relative path.
            # The code usually hardcodes the filename. Let's make sure it runs in the right dir or uses absolute path.
            # We'll try to patch the code to use the absolute path if it uses the basename.
            # Actually, simpler to just run it.
            
            execution_result = self._execute_code(code, image_path)
            
            return {
                "generated_code": code,
                "axis_ranges": axis_ranges,
                "execution_result": execution_result, # The parsed JSON data
                "raw_response": response
            }
        except Exception as e:
            return {"error": str(e)}

    def _execute_code(self, code, image_path):
        """
        Executes the generated code in a safe subprocess and captures stdout.
        """
        import subprocess
        import tempfile
        
        # Patch code to use full image path if it relies on local valid path
        # Or simple: write to a temp file in the SAME directory as the image? 
        # No, write to temp dir.
        
        # If code loads image using just basename, we might fail if CWD is different.
        # Let's replace the filename with the absolute path in the code.
        basename = os.path.basename(image_path)
        code = code.replace(f"'{basename}'", f"r'{image_path}'")
        code = code.replace(f'"{basename}"', f"r'{image_path}'")
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            # Run
            result = subprocess.run(['python3', tmp_path], capture_output=True, text=True, timeout=30)
            
            os.remove(tmp_path)
            
            if result.returncode != 0:
                print(f"      [Example Error Output]: {result.stderr[:200]}...")
                return {"error": "Execution failed", "stderr": result.stderr}
            
            # Parse STDOUT as JSON
            # stdout might contain other print statements. Look for the last JSON object.
            output_str = result.stdout.strip()
            # Try to find JSON block
            try:
                # Naive: try parsing whole string
                return json.loads(output_str)
            except:
                # Regex for json
                match = re.search(r'(\{.*\})', output_str, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                else:
                    return {"error": "No JSON found in stdout", "stdout": output_str}

        except Exception as e:
            return {"error": f"Execution wrapper failed: {e}"}

    def _clean_json_text(self, text):
        # Remove ```json ... ```
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()

    def _extract_code_block(self, text):
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text # Fallback
