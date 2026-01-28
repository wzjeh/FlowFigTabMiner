import os
import json
import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import TableTransformerForObjectDetection, AutoImageProcessor

def main():
    input_dir = "data/table-bodies-for-tara"
    output_json = os.path.join(input_dir, "tara_predictions.json")
    
    # Model ID
    model_name = "microsoft/table-transformer-structure-recognition-v1.1-all"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading TATR model: {model_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = TableTransformerForObjectDetection.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # COCO Data Structure
    coco_output = {
        "info": {
            "description": "TATR Pre-annotations for FlowFigTabMiner",
            "year": 2026,
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Categories
    # Get from model config
    id2label = model.config.id2label
    # COCO categories usually start at 1, but model output might be 0-indexed or mapped.
    # Let's map model_cls_id -> coco_category_id
    # We will just preserve the model's mapping but ensure it's recorded
    
    for cls_id, label in id2label.items():
        coco_output["categories"].append({
            "id": int(cls_id),
            "name": label,
            "supercategory": "table"
        })
        
    print(f"Categories: {[c['name'] for c in coco_output['categories']]}")

    images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not images:
        print("No images found!")
        return
        
    annotation_id = 1
    
    print(f"Processing {len(images)} images...")
    
    for img_id, img_path in enumerate(tqdm(images)):
        file_name = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            
            # Add Image info
            coco_output["images"].append({
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": file_name
            })
            
            # Inference
            # Using valid size to avoid crash as discovered previously
            inputs = processor(images=image, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1333}).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Convert [x1, y1, x2, y2] to COCO [x, y, w, h]
                x1, y1, x2, y2 = box.tolist()
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    continue
                    
                area = width * height
                
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, width, height],
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                    "score": float(score.item()) # Optional, helpful for filtering
                })
                annotation_id += 1
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    # Save JSON
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)
        
    print(f"Done! Saved COCO annotations to {output_json}")
    print(f"Total Annotations: {len(coco_output['annotations'])}")
    print("Instruction: In Roboflow, upload the images from 'data/table-bodies-for-tara' AND this 'tara_predictions.json' file together.")

if __name__ == "__main__":
    main()
