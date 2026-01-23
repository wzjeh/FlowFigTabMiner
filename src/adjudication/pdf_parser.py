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
        
        # Save cache
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
            
        return combined_text
