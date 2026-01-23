import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class CellClassifier:
    def __init__(self, model_path="models/cell_classifier_resnet.pt"):
        """
        Initialize the ResNet18 cell classifier.
        Args:
            model_path (str): Path to the fine-tuned ResNet18 weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.classes = ['Text', 'Structure'] # 0: Text, 1: Structure (example mapping)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading Cell Classifier (ResNet18) from {self.model_path}...")
        try:
            model = models.resnet18(weights=None) # Load structure
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2) # Binary classification
            
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                print(f"Warning: Cell Classifier weights not found at {self.model_path}. Using random weights (for dev/mock).")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading Cell Classifier: {e}")
            return None

    def classify_cells(self, cell_images):
        """
        Classify a list of cell images (PIL Images or paths).
        Returns:
            list[str]: list of class labels ("Text" or "Structure")
        """
        if self.model is None:
            return ["Text"] * len(cell_images) # Fallback

        results = []
        batch_tensors = []
        valid_indices = []

        # Preprocess batch
        for i, img_input in enumerate(cell_images):
            try:
                if isinstance(img_input, str):
                    image = Image.open(img_input).convert("RGB")
                else:
                    image = img_input.convert("RGB")
                
                input_tensor = self.transform(image)
                batch_tensors.append(input_tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error processing cell {i}: {e}")
        
        if not batch_tensors:
            return ["Text"] * len(cell_images)

        input_batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_batch)
            _, preds = torch.max(outputs, 1)
        
        # Map back to full list
        full_results = ["Error"] * len(cell_images)
        for idx, pred_idx in zip(valid_indices, preds):
            class_name = self.classes[pred_idx.item()]
            full_results[idx] = class_name
            
        return full_results
