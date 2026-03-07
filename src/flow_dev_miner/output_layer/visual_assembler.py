import os
import json
import logging
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VisualAssembler:
    """
    Creates visual verification artifacts by stitching the source figure/table image
    together with its extracted JSON data natively drawn as an image.
    """
    def __init__(self, output_dir="data/final_output/visuals"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Attempt to load a default font, fallback to basic PIL font
        try:
             self.font = ImageFont.truetype("arial.ttf", 14)
             self.title_font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
             self.font = ImageFont.load_default()
             self.title_font = ImageFont.load_default()

    def create_verification_image(self, original_image_path: str, item_json: dict, basename: str):
        """
        Creates a side-by-side or stacked image.
        """
        try:
            # 1. Load Original Image
            if not original_image_path or not os.path.exists(original_image_path):
                logger.warning(f"Could not find source image for {basename}: {original_image_path}")
                return None
                
            source_img = Image.open(original_image_path)
            if source_img.mode != 'RGB':
                source_img = source_img.convert('RGB')
                
            # Resize image if it's too huge
            max_img_width = 800
            if source_img.width > max_img_width:
                ratio = max_img_width / source_img.width
                new_h = int(source_img.height * ratio)
                source_img = source_img.resize((max_img_width, new_h), Image.Resampling.LANCZOS)
                
            # 2. Render JSON to Text Block
            text_lines = []
            text_lines.append(f"SOURCE: {basename}")
            text_lines.append(f"TYPE: {item_json.get('source_type', 'Unknown').upper()}")
            text_lines.append("-" * 40)
            
            # Key Semantic Data
            important_keys = ['Subfigure_Label', 'Catalyst', 'Solvent', 'Temperature_C', 'Pressure_MPa', 'Time']
            for k in important_keys:
                if k in item_json and item_json[k]:
                     text_lines.append(f"{k}: {item_json[k]}")
                     
            # Molecules & Yields
            mols = []
            for k, v in item_json.items():
                if k.startswith('Molecule_'):
                    mols.append(f"{k}: {v}")
            if mols:
                text_lines.append("-" * 40)
                text_lines.extend(mols)
            
            yields = []
            for k, v in item_json.items():
                if k.startswith('Yield_'):
                    yields.append(f"{k}: {v}")
            if yields:
                text_lines.append("-" * 40)
                text_lines.extend(yields)

            # Determine layout (Vertical stack is easiest)
            img_w, img_h = source_img.size
            
            # Calculate Text Block Size
            line_height = 20
            text_h = len(text_lines) * line_height + 40
            text_w = max(400, img_w) # Minimum width for text readability
            
            final_w = max(img_w, text_w)
            final_h = img_h + text_h
            
            # Combine
            final_img = Image.new('RGB', (final_w, final_h), color=(240, 240, 240))
            
            # Paste Image (centered horizontally if text is wider)
            img_offset_x = (final_w - img_w) // 2
            final_img.paste(source_img, (img_offset_x, 0))
            
            # Draw Text
            draw = ImageDraw.Draw(final_img)
            text_y = img_h + 20
            
            for line in text_lines:
                 draw.text((20, text_y), line, fill=(0, 0, 0), font=self.font)
                 text_y += line_height
                 
            # 3. Save
            out_path = os.path.join(self.output_dir, f"{basename}_verification.png")
            final_img.save(out_path)
            logger.info(f"Visual verification created: {out_path}")
            return out_path
            
        except Exception as e:
            logger.error(f"Failed to create verification image: {e}")
            return None
