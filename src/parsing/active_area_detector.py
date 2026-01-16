import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import pypdfium2 as pdfium
import json

class ActiveAreaDetector:
    def __init__(self, model_id="yifeihu/TF-ID-base"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
             self.device = "mps"
        
        print(f"Loading TF-ID (Florence-2) model: {model_id} on {self.device}...")
        # Florence-2 uses AutoProcessor and AutoModelForCausalLM
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Added attn_implementation="eager" to fix _supports_sdpa error
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            attn_implementation="eager"
        ).to(self.device).eval()

        # --- Compatibility Patch for Transformers > 4.40 ---
        # The remote code 'Florence2LanguageForConditionalGeneration' assumes it inherits from GenerationMixin
        # but PreTrainedModel no longer does in newer transformers versions. We inject it into the bases.
        if hasattr(self.model, "language_model"):
            target_class = self.model.language_model.__class__
            from transformers.generation import GenerationMixin, GenerationConfig
            
            # 1. Fix Inheritance
            if GenerationMixin not in target_class.__bases__:
                print(f"PATCHING: Adding GenerationMixin to {target_class.__name__} bases for compatibility.")
                target_class.__bases__ = (GenerationMixin,) + target_class.__bases__
            
            # 2. Fix generation_config (AttributeError: 'NoneType' object has no attribute '_from_model_config')
            if not hasattr(self.model.language_model, "generation_config") or self.model.language_model.generation_config is None:
                print("PATCHING: Initializing missing generation_config for language_model.")
                self.model.language_model.generation_config = GenerationConfig.from_model_config(self.model.language_model.config)
            
            # 3. Fix prepare_inputs_for_generation for DynamicCache (New Transformers)
            # The model expects past_key_values to be a tuple, but newer transformers return a Cache object or None in some places.
            original_prepare_inputs = self.model.language_model.prepare_inputs_for_generation
            def patched_prepare_inputs(decoder_input_ids, past_key_values=None, **kwargs):
                if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
                    # Assume it's a DynamicCache or similar and try to access legacy_cache or just pass it if the model handles it (it doesn't seems so)
                    # For Florence-2 specifically, it tries to access index [0][0].shape.
                    # If it's a Cache object, we might need to convert it or mock the access.
                    # However, simpler fix: if past_key_values is not standard, we might want to force use_cache=False in generation if possible,
                    # OR more robustly: patch the method to handle the specific crash line.
                    pass 
                return original_prepare_inputs(decoder_input_ids, past_key_values=past_key_values, **kwargs)

            # More direct fix: The crash is `past_length = past_key_values[0][0].shape[2]`.
            # If past_key_values is a DynamicCache, it doesn't support indexing like that.
            # We force use_cache=False for this specific debug/inference usage to avoid KV cache incompatibilities completely.
            self.model.language_model.generation_config.use_cache = False  # DISABLE KV CACHE to bypass structure mismatch


    def detect_tables_figures(self, image: Image.Image):
        """
        Detects tables and figures using Florence-2 <OD> task.
        """
        task_prompt = "<OD>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process generation to get bounding boxes
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        # Format results: parsed_answer is usually {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', ...]}}
        # But for simple <OD>, it might just be bboxes and labels
        detections = []
        if "<OD>" in parsed_answer:
            data = parsed_answer["<OD>"]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])
            
            for box, label in zip(bboxes, labels):
                detections.append({
                    "label": label,
                    "score": 1.0, # Florence-2 generation doesn't provide confidence scores easily in this mode
                    "box": [round(b, 2) for b in box]
                })
                
        return detections

    def process_pdf(self, pdf_path: str):
        """
        Renders PDF pages to images and detects objects on each page.
        """
        pdf = pdfium.PdfDocument(pdf_path)
        all_detections = {}
        
        try:
            for i, page in enumerate(pdf):
                # Render page to image
                bitmap = page.render(scale=2) # 2x scale
                pil_image = bitmap.to_pil()
                
                # Detect
                detections = self.detect_tables_figures(pil_image)
                if detections:
                    all_detections[f"page_{i+1}"] = detections
        finally:
            if pdf:
                pdf.close()
        
        return all_detections

    def save_crops(self, pdf_path: str, all_detections: dict, output_dir: str):
        """
        Crops detected regions from the PDF and saves them as images.
        Organized by Document Name -> tables/figures
        """
        pdf = pdfium.PdfDocument(pdf_path)
        
        # Extract document name for subfolder
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        doc_output_dir = os.path.join(output_dir, doc_name)
        
        os.makedirs(doc_output_dir, exist_ok=True)
        
        saved_paths = []
        
        try:
            for page_key, detections in all_detections.items():
                page_idx = int(page_key.split("_")[1]) - 1
                page = pdf[page_idx]
                # Render page to image at high resolution for cropping
                bitmap = page.render(scale=3) # 3x scale for better crop quality
                pil_image = bitmap.to_pil()
                # 3x scale vs 2x scale detection
                scale_factor = 3.0 / 2.0
                
                for idx, item in enumerate(detections):
                    label = item["label"] # 'table' or 'figure'
                    box = item["box"] # [x1, y1, x2, y2]
                    
                    # Scale box
                    crop_box = [coord * scale_factor for coord in box]
                    
                    # Label directory: data/intermediate/doc_name/tables/
                    label_dir = os.path.join(doc_output_dir, f"{label}s") 
                    os.makedirs(label_dir, exist_ok=True)
                    
                    # Crop and Save
                    crop = pil_image.crop(crop_box)
                    filename = f"{page_key}_{label}_{idx}.png"
                    filepath = os.path.join(label_dir, filename)
                    crop.save(filepath)
                    saved_paths.append(filepath)
        finally:
            if pdf:
                pdf.close()
                
        return saved_paths

if __name__ == "__main__":
    # Test script
    import sys
    import json
    if len(sys.argv) > 1:
        detector = ActiveAreaDetector()
        pdf_path = sys.argv[1]
        results = detector.process_pdf(pdf_path)
        print(json.dumps(results, indent=2))
        
        # Test saving
        output_dir = "data/intermediate"
        print(f"Saving crops to {output_dir}...")
        detector.save_crops(pdf_path, results, output_dir)
