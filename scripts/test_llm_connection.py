import os
import sys

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.adjudication.llm_engine import LLMEngine
from dotenv import load_dotenv

def main():
    print("--- Testing LLM Connection ---")
    load_dotenv(override=True)
    
    print(f"Environment LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"Environment LLM_MODEL_NAME: {os.getenv('LLM_MODEL_NAME')}")
    
    try:
        engine = LLMEngine()
        print(f"Initialized LLMEngine with Provider: {engine.provider}")
        print(f"Model: {engine.model}")
        
        if engine.provider == "dashscope":
            import dashscope
            print(f"DashScope API Key set: {'Yes' if dashscope.api_key else 'No'}")
            print(f"DashScope Base URL: {dashscope.base_http_api_url}")

        print("\nSending test message...")
        response = engine.chat("You are a test bot.", "Hello! Are you online? Reply with 'Yes, I am online as [Model Name]'.")
        
        print("\n--- Response ---")
        print(response)
        
        if "Error" in response:
            print("\n[FAIL] Connection failed.")
        else:
            print("\n[SUCCESS] Connection verified.")
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
