import os
import fitz  # PyMuPDF

class PDFParser:
    def __init__(self):
        pass

    def extract_text(self, pdf_path):
        """
        Extracts full text from a PDF.
        Caches the result to a .txt file in the same directory.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        # Check cache
        txt_path = pdf_path.replace(".pdf", "_fulltext.txt")
        if os.path.exists(txt_path):
            print(f"[PDFParser] Loading cached text from {txt_path}")
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
                
        # Parse PDF
        print(f"[PDFParser] Parsing {pdf_path}...")
        full_text = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                full_text.append(text)
            doc.close()
        except Exception as e:
            print(f"[PDFParser] Error parsing PDF: {e}")
            return ""
            
        combined_text = "\n".join(full_text)
        
        # Truncate
        combined_text = self._truncate_text(combined_text)
        
        # Save cache
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
            
        return combined_text

    def _truncate_text(self, text):
        """
        Truncate text after References/Bibliography/Conclusion to save tokens.
        """
        # Lowercase for search
        text_lower = text.lower()
        
        # Keywords to cut off
        # "references" often appears in headers, need to be careful?
        # Usually looking for a section header.
        # Simple heuristic: Split by newlines and look for standalone lines?
        # Or just find the last occurrence?
        # Often "References" is at the end. 
        
        cutoff_keywords = ["references", "bibliography", "acknowledgements"]
        
        cutoff_idx = len(text)
        
        for kw in cutoff_keywords:
            # We want headers, perhaps "\nReferences" or "\nREFERENCES"
            # Try to find the last substantial block or just the first occurrence from the end?
            # It's tricky. Let's try finding "\nReferences\n"
            
            # Simple approach: Find last 20% of text? No.
            # Just find the keyword.
            idx = text_lower.rfind(kw)
            if idx != -1 and idx < cutoff_idx:
                # Basic check: is it really near the end?
                # If it's in the first 10%, it's probably citation like "see References...".
                if idx > len(text) * 0.5:
                     cutoff_idx = idx
        
        if cutoff_idx < len(text):
            print(f"[PDFParser] Truncated text at index {cutoff_idx} (detected Reference/Biblio).")
            return text[:cutoff_idx]
            
        return text
