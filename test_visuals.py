import json
import os
import sys

sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.output_layer.visual_assembler import VisualAssembler

def run():
    print("Testing VisualAssembler on existing example_final_summary.json", flush=True)
    with open("data/final_output/example_final_summary.json", 'r') as f:
        data = json.load(f)
        
    visualizer = VisualAssembler(output_dir="data/final_output/example_visuals")
    for i, row in enumerate(data):
        sk = row.get('_source_key', '')
        if not sk:
            continue
            
        row_id = f"{sk}_row_{i}"
        if "table" in sk:
            orig_img = f"data/intermediate/example/{sk}/{sk}_body_main.png"
        else:
            orig_img = f"data/intermediate/example/macro_cleaned/{sk}_cleaned.png"
        
        if os.path.exists(orig_img):
            v_path = visualizer.create_verification_image(orig_img, row, row_id)
            print(f"Generated: {v_path}")
        else:
            print(f"Could not find image: {orig_img}")
            
if __name__ == '__main__':
    run()
