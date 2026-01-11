import json
import os

nb_path = "debug_pipeline.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Code for Cell 4 (Table Extraction)
new_table_source = [
    "# 4. Table Extraction (Step 3A)\n",
    "from src.extraction.table_agent import extract_table\n",
    "import os\n",
    "\n",
    "extracted_tables = []\n",
    "# Use the cropped images from Step 3\n",
    "if 'table_images' in locals() and table_images:\n",
    "    print(f\"Processing {len(table_images)} Tables...\")\n",
    "    for img_path in table_images:\n",
    "        print(f\"--> Extracting: {os.path.basename(img_path)}\")\n",
    "        try:\n",
    "            result = extract_table(img_path)\n",
    "            if result.get(\"is_valid\"):\n",
    "                print(\"    ✅ Valid Table Extracted\")\n",
    "                extracted_tables.append({\"source\": img_path, \"data\": result})\n",
    "            else:\n",
    "                print(f\"    ❌ Rejected: {result.get('reason')}\")\n",
    "        except Exception as e:\n",
    "            print(f\"    ⚠️ Error: {e}\")\n",
    "else:\n",
    "    print(\"No table images found from Step 3.\")\n",
    "\n",
    "print(f\"\\nTotal Valid Tables: {len(extracted_tables)}\")"
]

# Code for Cell 5 (Figure Extraction)
new_figure_source = [
    "# 5. Figure Extraction (Step 3B)\n",
    "from src.extraction.figure_agent import extract_figure\n",
    "import os\n",
    "\n",
    "extracted_figures = []\n",
    "if 'figure_images' in locals() and figure_images:\n",
    "    print(f\"Processing {len(figure_images)} Figures...\")\n",
    "    for img_path in figure_images:\n",
    "        print(f\"--> Extracting: {os.path.basename(img_path)}\")\n",
    "        try:\n",
    "            result = extract_figure(img_path)\n",
    "            if result.get(\"is_valid\"):\n",
    "                print(\"    ✅ Valid Figure Extracted\")\n",
    "                extracted_figures.append({\"source\": img_path, \"data\": result})\n",
    "            else:\n",
    "                print(f\"    ❌ Rejected: {result.get('reason')}\")\n",
    "        except Exception as e:\n",
    "            print(f\"    ⚠️ Error: {e}\")\n",
    "else:\n",
    "    print(\"No figure images found from Step 3.\")\n",
    "\n",
    "print(f\"\\nTotal Valid Figures: {len(extracted_figures)}\")"
]

# Update extracted logic
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Identify Table Cell
        if "# 4. Table Extraction" in source_str or "Testing Table Agent" in source_str:
            print("Updating Table Extraction Cell...")
            cell['source'] = new_table_source
            cell['outputs'] = [] # Clear output

        # Identify Figure Cell
        if "# 5. Figure Extraction" in source_str or "Testing Figure Agent" in source_str:
            print("Updating Figure Extraction Cell...")
            cell['source'] = new_figure_source
            cell['outputs'] = [] # Clear output

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
