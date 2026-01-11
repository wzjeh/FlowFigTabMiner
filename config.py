import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Project Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
    DATA_INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro")

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")

    @classmethod
    def validate(cls):
        """Validates that necessary configuration is present."""
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not found in environment variables.")
        if cls.LLM_PROVIDER == "qwen" and not cls.QWEN_API_KEY:
            print("WARNING: QWEN_API_KEY not found in environment variables.")
        # Add basic directory validation
        os.makedirs(cls.DATA_INPUT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_INTERMEDIATE_DIR, exist_ok=True)
        os.makedirs(cls.DATA_OUTPUT_DIR, exist_ok=True)

# Run validation on import
Config.validate()
