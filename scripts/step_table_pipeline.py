import sys
import os
import argparse
import json
import base64

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.table_pipeline import TablePipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to table image")
    parser.add_argument("--output_dir", help="Output directory", default=None)
    args = parser.parse_args()

    pipeline = TablePipeline()
    result = pipeline.process_table(args.image_path, output_dir=args.output_dir)
    
    # Serialize for stdout
    # Dataframe not serializable directly, convert to dict
    if 'dataframe' in result:
        result['dataframe'] = result['dataframe'].to_dict(orient='records')
    
    # Structure/Cells might have float32 from numpy/torch
    # Recursively clean? Or assume json.dump handles basics or use default encoder
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return super().default(obj)
    
    print("---JSON_START---")
    print(json.dumps(result, cls=NumpyEncoder))
    print("---JSON_END---")

if __name__ == "__main__":
    main()
