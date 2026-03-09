import os
import sys
import glob

sys.path.insert(0, os.getcwd())
from src.adjudication.pdf_parser import PDFParser

def test_truncation(directory, max_files=None):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]
        
    print(f"\nTesting {len(pdf_files)} files in {directory}...")
    
    parser = PDFParser()
    fails = []
    
    for pdf in pdf_files:
        # force re-extraction by removing txt cache if exists
        txt_path = pdf.replace(".pdf", "_fulltext.txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
            
def test_truncation(directory, max_files=None):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]
        
    print(f"\nTesting {len(pdf_files)} files in {directory}...")
    
    parser = PDFParser()
    fails = []
    
    for pdf in pdf_files:
        # force re-extraction by removing txt cache if exists
        txt_path = pdf.replace(".pdf", "_fulltext.txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
            
        text = parser.extract_text(pdf)
        
        # Check if references snuck through
        text_lower = text.lower()
        min_pos = int(len(text) * 0.6)
        
        # very simple check to see if "references" or "bibliography" exist in the last part
        has_ref = "\nreferences\n" in text_lower[min_pos:] or "\nbibliography\n" in text_lower[min_pos:] or " references\n" in text_lower[min_pos:] or "bibliography" in text_lower[-1000:]
        
        if has_ref:
            fails.append(pdf)
            print(f" -> Testing {os.path.basename(pdf)} -> [FAIL]")
        else:
            # print(f" -> Testing {os.path.basename(pdf)} -> [OK]")
            pass

    return fails

def main():
    dirs = [
        "data/input/补充",
        "data/input/普通"
    ]
    
    total_fails = []
    for d in dirs:
        fails = test_truncation(d, max_files=None) 
        total_fails.extend(fails)
        
    if total_fails:
        print(f"\nFailed to truncate references in {len(total_fails)} files:")
        for f in total_fails:
            print(f"  - {f}")
    else:
        print("\nAll tested files successfully truncated references!")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
