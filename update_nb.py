import json
import os

nb_path = "debug_pipeline.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new source code for the first code cell (Step 1)
new_source = [
    "# 1. Setup & Configuration\n",
    "import os\n",
    "import sys\n",
    "\n",
    "# Ensure project root is in path\n",
    "sys.path.append(os.getcwd())\n",
    "\n",
    "from config import Config\n",
    "from src.llm_factory import LLMFactory\n",
    "\n",
    "# Note: All configuration is now centrally managed by src.config.Config and .env\n",
    "# This ensures consistency between this notebook and main.py\n",
    "\n",
    "print(f\"Current Provider: {Config.LLM_PROVIDER}\")\n",
    "print(f\"Current Model: {Config.LLM_MODEL_NAME}\")\n",
    "\n",
    "# Test LLM Connection\n",
    "try:\n",
    "    print(\"Testing LLM connection...\")\n",
    "    # response = LLMFactory.create_completion(\"Hello, config is working!\")\n",
    "    # print(f\"LLM Response: {response}\")\n",
    "    print(\"LLM Factory initialized (Uncomment lines above to test actual call)\")\n",
    "except Exception as e:\n",
    "    print(f\"LLM Connection Error: {e}\")"
]

# Find the cell to update. It's the first code cell.
# In the provided view, it was index 1 (index 0 was markdown).
# We can check the source to be sure.
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check if it looks like the setup cell
        source_str = "".join(cell['source'])
        if "1. Setup" in source_str or "manual overrides" in source_str.lower():
            print("Found target cell. Updating...")
            cell['source'] = new_source
            cell['outputs'] = [] # Clear outputs to be clean
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
