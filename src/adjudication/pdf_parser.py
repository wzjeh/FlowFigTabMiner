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
        # We look for section headers that typically appear at the end.
        # "conclusions" or "conclusion" is often the last section before refs.
        # "references" is the definitive end.
        
        cutoff_keywords = [
            "\nreferences", "\nbibliography", "\nacknowledgements", 
            "\nconclusion", "\nconclusions", "experimental section"
        ]
        
        cutoff_idx = len(text)
        
        # We want to find the *first* occurrence of these *after* the middle of the document
        # to avoid matching "See references" in the intro.
        
        min_pos = len(text) * 0.6 
        
        for kw in cutoff_keywords:
            idx = text_lower.find(kw, int(min_pos))
            if idx != -1:
                # We found a keyword near the end.
                # Use the earliest one found (e.g. Conclusion comes before References)
                if idx < cutoff_idx:
                    cutoff_idx = idx
        
        if cutoff_idx < len(text):
            print(f"[PDFParser] Truncated text at index {cutoff_idx}/{len(text)} (detected ending section).")
            return text[:cutoff_idx]
            
        return text
