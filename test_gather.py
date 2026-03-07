import os
import json

def gather_evidence_items(intermediate_dir, pdf_id):
    evidence_items = []
    if os.path.exists(intermediate_dir):
        for root, dirs, files in os.walk(intermediate_dir):
            for file in files:
                if file.endswith("_evidence.json"):
                    path = os.path.join(root, file)
                    print(f"Found candidate: {path}")
                    try:
                        with open(path, 'r') as fp:
                            data = json.load(fp)
                            rel = data.get('is_relevant', False)
                            print(f"  -> is_relevant: {rel}")
                            if rel:
                                evidence_items.append(path)
                    except Exception as e:
                        print(f"  -> Error: {e}")
    return evidence_items

items = gather_evidence_items("data/intermediate/example", "example")
print(f"Final gathered: {items}")
