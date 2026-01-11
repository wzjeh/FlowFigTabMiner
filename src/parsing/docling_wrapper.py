from docling.document_converter import DocumentConverter

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Parses a PDF file and returns its content as Markdown.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        
    Returns:
        str: The extracted content in Markdown format.
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    # Export to markdown
    return result.document.export_to_markdown()

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        print(parse_pdf_to_markdown(sys.argv[1]))
