import os
import argparse
import json
import time
from src.parsing.docling_wrapper import parse_pdf_to_markdown
from src.parsing.active_area_detector import ActiveAreaDetector
from src.extraction.table_agent import extract_table
from src.extraction.figure_agent import extract_figure
# from src.fusion.data_merger import fuse_data 

def main():
    parser = argparse.ArgumentParser(description="FlowFigTabMiner: Extract data from Flow Chemistry Papers")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--output_dir", default="data/output", help="Directory to save final results")
    
    args = parser.parse_args()
    input_pdf = args.pdf_path
    
    if not os.path.exists(input_pdf):
        print(f"Error: File not found: {input_pdf}")
        return

    print(f"--- Starting Pipeline for {input_pdf} ---")
    start_time = time.time()

    # Step 1: Parsing (Docling) - Optional for now or acting as context
    # print("Step 1: Parsing text with Docling...")
    # try:
    #     markdown_text = parse_pdf_to_markdown(input_pdf)
    #     print(f"   Parsed {len(markdown_text)} characters of text.")
    # except Exception as e:
    #     print(f"   Warning: Docling parsing failed: {e}")
    #     markdown_text = ""
    
    # Step 2: TF-ID Detection & Cropping
    print("Step 2: Detecting Tables and Figures (TF-ID)...")
    try:
        detector = ActiveAreaDetector()
        detections = detector.process_pdf(input_pdf)
        
        intermediate_dir = "data/intermediate"
        print(f"   Saving crops to {intermediate_dir}...")
        saved_paths = detector.save_crops(input_pdf, detections, intermediate_dir)
        
        table_images = [p for p in saved_paths if '_table_' in p]
        figure_images = [p for p in saved_paths if '_figure_' in p]
        print(f"   Found {len(table_images)} tables and {len(figure_images)} figures.")
    except Exception as e:
        print(f"Error during TF-ID step: {e}")
        return

    # Step 3: Extraction
    print("Step 3: Extracting Data using LLM...")
    
    extracted_data = {
        "metadata": {
            "source_pdf": input_pdf,
            "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "tables": [],
        "figures": []
    }

    # 3A: Tables
    if table_images:
        print(f"   Processing {len(table_images)} Tables...")
        for img_path in table_images:
            print(f"      extracting: {os.path.basename(img_path)}")
            try:
                table_result = extract_table(img_path)
                if table_result.get("is_valid"):
                    extracted_data["tables"].append({
                        "source_image": img_path,
                        "data": table_result
                    })
                else:
                    print(f"         -> Rejected ({table_result.get('reason', 'unknown')})")
            except Exception as e:
                print(f"         -> Error: {e}")
    else:
        print("   No tables to process.")

    # 3B: Figures
    if figure_images:
        print(f"   Processing {len(figure_images)} Figures...")
        for img_path in figure_images:
            print(f"      extracting: {os.path.basename(img_path)}")
            try:
                fig_result = extract_figure(img_path)
                if fig_result.get("is_valid"):
                    extracted_data["figures"].append({
                        "source_image": img_path,
                        "data": fig_result
                    })
                else:
                    print(f"         -> Rejected ({fig_result.get('reason', 'unknown')})")
            except Exception as e:
                print(f"         -> Error: {e}")
    else:
        print("   No figures to process.")
            
    # Step 4: Fusion (Placeholder)
    # merged_data = fuse_data(...)
    
    # Save Final Output
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_pdf))[0]
    output_json = os.path.join(args.output_dir, f"{basename}_results.json")
    
    with open(output_json, "w") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"--- Pipeline Complete in {elapsed:.1f}s! Results saved to {output_json} ---")

if __name__ == "__main__":
    main()
