import google.generativeai as genai
import os

class LLMFactory:
    @staticmethod
    def get_model():
        # Lazy import to ensure we pick up the latest reloaded config
        from config import Config
        
        provider = Config.LLM_PROVIDER
        model_name = Config.LLM_MODEL_NAME

        if provider == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found.")
            genai.configure(api_key=Config.GEMINI_API_KEY)
            return genai.GenerativeModel(model_name)
        
        elif provider == "qwen":
            import dashscope
            if not Config.QWEN_API_KEY:
                raise ValueError("QWEN_API_KEY not found.")
            dashscope.api_key = Config.QWEN_API_KEY
            # CRITICAL: Set International Endpoint used in successful test
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
            return dashscope
        
        else:
            raise ValueError(f"Unknown LLM Provider: {provider}")

    @staticmethod
    def create_completion(prompt, image_path=None):
        """
        Unified interface for simple text/image-to-text generation.
        """
        # Lazy import config
        from config import Config
        
        provider = Config.LLM_PROVIDER
        model_name = Config.LLM_MODEL_NAME
        
        if provider == "gemini":
            try:
                model = LLMFactory.get_model()
                inputs = [prompt]
                if image_path:
                    # Load image for Gemini
                    import PIL.Image
                    img = PIL.Image.open(image_path)
                    inputs.append(img)
                
                response = model.generate_content(inputs)
                return response.text
            except Exception as e:
                print(f"Gemini Error: {e}")
                return None
            
        elif provider == "qwen":
            import dashscope
            from http import HTTPStatus
            import base64
            
            LLMFactory.get_model() # Ensure config is set
            
            content = []
            
            # 1. Handle Image via Base64 Data URI (Bypasses SDK Upload Bug)
            if image_path:
                try:
                    with open(image_path, "rb") as img_file:
                        b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    mime_type = "image/jpeg"
                    if image_path.lower().endswith(".png"):
                        mime_type = "image/png"
                    elif image_path.lower().endswith(".webp"):
                        mime_type = "image/webp"
                        
                    data_uri = f"data:{mime_type};base64,{b64_data}"
                    content.append({"image": data_uri})
                except Exception as e:
                    print(f"Image Encoding Error: {e}")
                    return None
            
            # 2. Add Prompt
            content.append({"text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            try:
                # Use MultiModalConversation
                response = dashscope.MultiModalConversation.call(
                    model=model_name, # e.g., 'qwen-vl-max'
                    messages=messages
                )
                
                if response.status_code == HTTPStatus.OK:
                    # Parse Qwen VL response which is usually a list of dicts: [{'text': '...'}]
                    output_content = response.output.choices[0].message.content
                    if isinstance(output_content, list):
                        # Extract all text parts
                        return " ".join([item["text"] for item in output_content if "text" in item])
                    else:
                        return str(output_content)
                else:
                    print(f"Qwen Error: {response.code} - {response.message}")
                    return None
            except Exception as e:
                print(f"Qwen Exception: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return None
