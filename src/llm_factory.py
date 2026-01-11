import google.generativeai as genai
# import dashscope # Uncomment when using Qwen via Dashscope
import os
from config import Config

class LLMFactory:
    @staticmethod
    def get_model():
        provider = Config.LLM_PROVIDER
        model_name = Config.LLM_MODEL_NAME

        if provider == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found.")
            genai.configure(api_key=Config.GEMINI_API_KEY)
            return genai.GenerativeModel(model_name)
        
        elif provider == "qwen":
            if not Config.QWEN_API_KEY:
                raise ValueError("QWEN_API_KEY not found.")
            # Placeholder for Qwen implementation
            # dashscope.api_key = Config.QWEN_API_KEY
            # return dashscope.Generation.call(model=model_name, ...)
            raise NotImplementedError("Qwen implementation pending.")
        
        else:
            raise ValueError(f"Unknown LLM Provider: {provider}")

    @staticmethod
    def create_completion(prompt, image_path=None):
        """
        Unified interface for simple text/image-to-text generation.
        """
        model = LLMFactory.get_model()
        provider = Config.LLM_PROVIDER
        
        if provider == "gemini":
            inputs = [prompt]
            if image_path:
                # Load image for Gemini
                import PIL.Image
                img = PIL.Image.open(image_path)
                inputs.append(img)
            
            response = model.generate_content(inputs)
            return response.text
            
        elif provider == "qwen":
             # Placeholder for Qwen
             pass
        
        return None
