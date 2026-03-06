import os
from PIL import Image, ImageDraw, ImageFont

def create_yolo_grid():
    # Model config
    models = [
        {
            "path": "models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train",
            "prefix": "fig-seg"
        },
        {
            "path": "models/yolo11m-fig-scatter-0208/runs/detect/train",
            "prefix": "fig-sca"
        },
        {
            "path": "models/yolo11m-table-seg-0208/runs/detect/train",
            "prefix": "tab-seg"
        },
        {
            "path": "models/yolo11s-tab-molecule-0207/runs/detect/train",
            "prefix": "tab-mol"
        }
    ]
    
    image_types = ["PR_curve.png", "F1_curve.png", "confusion_matrix_normalized.png"]
    target_height = 800  # Height for each row's subplots
    padding = 40        # Padding between subplots
    text_height = 60    # Space for caption below image
    
    rows = []
    global_index = 0 # Sequential index for (a), (b), ...
    
    for i, model in enumerate(models):
        row_images = []
        for j, img_type in enumerate(image_types):
            img_path = os.path.join(model["path"], img_type)
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found.")
                continue
            
            img = Image.open(img_path)
            # Resize while maintaining aspect ratio
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            # Create a canvas for image + text
            canvas = Image.new("RGB", (new_width, target_height + text_height), "white")
            canvas.paste(img, (0, 0))
            
            # Draw label
            draw = ImageDraw.Draw(canvas)
            label = f"({chr(97 + global_index)})"
            global_index += 1
            # Center text (approximation since font loading is tricky in different envs)
            # Use default font
            try:
                # Try to find a font, or fallback to default
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
            except:
                font = ImageFont.load_default()
            
            # Calculate position for label
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            draw.text(((new_width - text_w) // 2, target_height + (text_height - text_h) // 2 - 5), label, fill="black", font=font)
            
            row_images.append(canvas)
        
        # Combine row images
        total_row_width = sum(img.width for img in row_images) + padding * (len(row_images) - 1)
        row_canvas = Image.new("RGB", (total_row_width, target_height + text_height), "white")
        current_x = 0
        for img in row_images:
            row_canvas.paste(img, (current_x, 0))
            current_x += img.width + padding
        rows.append(row_canvas)
    
    # Final image assembly
    max_row_width = max(row.width for row in rows)
    total_height = sum(row.height for row in rows) + padding * (len(rows) - 1)
    
    final_img = Image.new("RGB", (max_row_width + padding * 2, total_height + padding * 2), "white")
    current_y = padding
    for row in rows:
        # Center the row if widths vary
        x_offset = (max_row_width + padding * 2 - row.width) // 2
        final_img.paste(row, (x_offset, current_y))
        current_y += row.height + padding
    
    # Save output
    output_dir = "pub/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig-yolo-results.png")
    final_img.save(output_path)
    print(f"Saved final image to {output_path}")

if __name__ == "__main__":
    create_yolo_grid()
