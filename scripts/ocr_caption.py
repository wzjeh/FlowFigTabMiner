import os
import sys
import json
from paddleocr import PaddleOCR

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extract_caption(image_path):
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return

    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(image_path, cls=True)

        caption_lines = []
        full_text = []
        
        if result and result[0]:
            # Sort by Y position (top to bottom)
            sorted_res = sorted(result[0], key=lambda x: x[0][0][1])
            
            for line in sorted_res:
                text = line[1][0]
                conf = line[1][1]
                full_text.append(text)
                
                # Heuristic: Captions usually start with "Fig" or "Figure" or "Table"
                # And usually are at the bottom (checked by logic or simply gathering all valid text)
                # For now, let's just grab lines starting with keywords, or if not found, grab the last few lines?
                # Better: Grab lines starting with "Fig" and subsequent lines until end?
                
                lower_text = text.lower()
                if lower_text.startswith("fig") or lower_text.startswith("scheme") or lower_text.startswith("table"):
                    caption_lines.append(text)
                elif caption_lines: # Append continuation lines
                    caption_lines.append(text)

        # If heuristic matched nothing, return empty string (or maybe full text if desperate?)
        # User specifically wants "Fig 2...", so heuristic is safe.
        final_caption = " ".join(caption_lines)
        
        print("---JSON_START---")
        print(json.dumps({"caption": final_caption, "full_text": full_text}))
        print("---JSON_END---")
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_caption(sys.argv[1])
