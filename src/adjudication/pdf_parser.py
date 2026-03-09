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
                content = f.read()
                # Re-truncate in case of old cache
                truncated = self._truncate_text(content)
                if len(truncated) < len(content):
                    with open(txt_path, "w", encoding="utf-8") as fw:
                        fw.write(truncated)
                    return truncated
                return content
                
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
        import re
        text_lower = text.lower()
        
        # We only consider the last 40% of the document to avoid false positives.
        min_pos = int(len(text) * 0.6)
        
        # We can use regex to find section headers robustly
        patterns = [
            r'\n\s*(?:[0-9]*\.?\s*)?references\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?bibliography\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?acknowledgements?\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?conclusions?\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?notes\s*and\s*references\s*\n'
        ]
        
        cutoff_idx = len(text)
        
        for pattern in patterns:
            match = re.search(pattern, text_lower[min_pos:])
            if match:
                idx = min_pos + match.start()
                if idx < cutoff_idx:
                    cutoff_idx = idx
                    
        # Fallback to simple rfind for references if regex misses
        if cutoff_idx == len(text):
            fallback_keywords = ["\nreferences", "\nacknowledgement"]
            for kw in fallback_keywords:
                idx = text_lower.rfind(kw)
                if idx > min_pos and idx < cutoff_idx:
                    cutoff_idx = idx
        
        if cutoff_idx < len(text):
            print(f"[PDFParser] Truncated text at index {cutoff_idx}/{len(text)} (detected ending section).")
            return text[:cutoff_idx]
            
        return text
