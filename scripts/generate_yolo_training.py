import os
from PIL import Image, ImageDraw, ImageFont

def create_training_stack():
    # Model config
    models = [
        "models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train",
        "models/yolo11m-fig-scatter-0208/runs/detect/train",
        "models/yolo11m-table-seg-0208/runs/detect/train",
        "models/yolo11s-tab-molecule-0207/runs/detect/train"
    ]
    
    label_text = ["(a)", "(b)", "(c)", "(d)"]
    padding = 40
    
    images = []
    for i, model_path in enumerate(models):
        img_path = os.path.join(model_path, "results.png")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found.")
            continue
        
        img = Image.open(img_path)
        
        # Draw label in top-right corner
        draw = ImageDraw.Draw(img)
        label = label_text[i]
        
        try:
            # Try to find a font, or fallback to default
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        except:
            font = ImageFont.load_default()
            
        # Get text bbox
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Position in top-left with a little margin
        margin = 20
        pos = (margin, margin)
        
        # Optional: draw white background for label if needed for contrast
        # draw.rectangle([pos[0]-5, pos[1]-5, pos[0]+text_w+5, pos[1]+text_h+5], fill="white")
        
        draw.text(pos, label, fill="black", font=font)
        images.append(img)
    
    if not images:
        print("No images found. Exiting.")
        return

    # Stack vertically
    total_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + padding * (len(images) - 1)
    
    final_img = Image.new("RGB", (total_width, total_height), "white")
    current_y = 0
    for img in images:
        # Center horizontally if widths vary (though YOLO results are usually uniform)
        x_offset = (total_width - img.width) // 2
        final_img.paste(img, (x_offset, current_y))
        current_y += img.height + padding
        
    # Save output
    output_dir = "pub/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig-yolo-training.png")
    final_img.save(output_path)
    print(f"Saved training stack to {output_path}")

if __name__ == "__main__":
    create_training_stack()
