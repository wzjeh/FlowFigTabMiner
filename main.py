import argparse
from config import Config
from src.llm_factory import LLMFactory

def main():
    parser = argparse.ArgumentParser(description="FlowFigTabMiner CLI")
    parser.add_argument("--input", "-i", type=str, help="Input PDF file path")
    args = parser.parse_args()

    print(f"FlowFigTabMiner initialized.")
    print(f"Provider: {Config.LLM_PROVIDER}, Model: {Config.LLM_MODEL_NAME}")
    
    if args.input:
        print(f"Processing: {args.input}")
        # Pipeline logic will go here
    else:
        print("No input file specified. Use --input to specify a PDF.")

if __name__ == "__main__":
    main()
