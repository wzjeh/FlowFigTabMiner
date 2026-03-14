import os
import json
import glob
from src.adjudication.llm_engine import LLMEngine
from src.adjudication.pdf_parser import PDFParser

class GlobalAssembly:
    def __init__(self, output_dir="data/final_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = LLMEngine()
        self.pdf_parser = PDFParser()

    def run(self, pdf_path, intermediate_dir=None, force=False):
        """
        Run global assembly for a PDF.
        If force=False and _final.json already exists, skip LLM call and return cached path.
        """
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        if not intermediate_dir:
            intermediate_dir = os.path.join("data/intermediate", basename)

        out_file = os.path.join(self.output_dir, f"{basename}_final.json")
        if not force and os.path.exists(out_file):
            print(f"[GlobalAssembly] Cache hit — skipping LLM (use --force-assembly to rerun): {out_file}")
            # Still run Excel export from cached JSON
            try:
                with open(out_file) as f:
                    records = json.load(f)
                if isinstance(records, list):
                    self._save_excel(records, basename)
            except Exception as e:
                print(f"[GlobalAssembly] Excel from cache skipped: {e}")
            return out_file

        print(f"[GlobalAssembly] Assembling final dataset for {basename}...")
        
        # 1. Get Text
        full_text = self.pdf_parser.extract_text(pdf_path)
        
        # 2. Get Figure Data
        # Evidence JSONs are usually in data/evidence, named {figure_id}_evidence.json
        # Figure IDs usually contain the pdf basename or page info?
        # The assembler saves them to data/evidence.
        # We need to filter for this PDF.
        # If Figure ID scheme is page_X_figure_Y, we might need a mapping.
        # Currently, FigurePipeline processes images in `data/intermediate/{basename}/figures`.
        # The generated JSONs are in `data/evidence`.
        # Problem: 'data/evidence' is a flat folder? 
        # Yes, based on scripts/run_steps2_to_4.py defaulting to EvidenceAssembler default.
        
        # We should look for JSONs that correspond to this PDF.
        # If FigurePipeline used `figure_id` derived from filename like `page_0_figure_0`, 
        # it doesn't strictly have the PDF basename in it unless we added it.
        # TF-ID naming convention: `page_{page_num}_figure_{fig_num}.png` inside `{basename}/figures`.
        # So the IDs are generic "page_0_figure_0". 
        # This implies `data/evidence` might mix evidence from different PDFs if we are not careful?
        # Or maybe we assume we run one PDF at a time and clear evidence?
        # Or we should check `meta['source_intermediate_dir']` inside JSONs?
        
        # Figure evidence JSONs are saved in data/intermediate/{basename}/macro_cleaned/
        figure_data = []
        macro_cleaned_dir = os.path.join(intermediate_dir, "macro_cleaned")
        for evidence_dir in [macro_cleaned_dir, "data/evidence"]:
            if os.path.exists(evidence_dir):
                for jpath in glob.glob(os.path.join(evidence_dir, "*_evidence.json")):
                    try:
                        with open(jpath, 'r') as f:
                            d = json.load(f)
                        meta = d.get('meta', {})
                        source_dir = meta.get('source_intermediate_dir', '')
                        if basename in source_dir or evidence_dir == macro_cleaned_dir:
                            figure_data.append(d)
                    except: pass

        print(f"   -> Found {len(figure_data)} figure evidence packets.")
        
        # 3. Get Table Data
        # TablePipeline saves CSVs and internal JSONs in `data/intermediate/{basename}/tables/`
        tables_dir = os.path.join(intermediate_dir, "tables")
        table_data = []
        if os.path.exists(tables_dir):
            # Recurse? Or just check subfolders
            # Logic: For each subfolder in tables_dir, look for .csv
            for root, dirs, files in os.walk(tables_dir):
                for file in files:
                    if file.endswith(".csv"):
                         csv_path = os.path.join(root, file)
                         # Load CSV content
                         with open(csv_path, 'r') as f:
                             csv_content = f.read()
                         table_data.append({
                             "table_name": file,
                             "content": csv_content
                         })
        
        print(f"   -> Found {len(table_data)} table data packets.")

        # 3.5. Load sub-variable libraries
        local_vars_map = {}
        local_vars_dir = os.path.join(intermediate_dir, "local_vars")
        if os.path.exists(local_vars_dir):
            for lv_path in glob.glob(os.path.join(local_vars_dir, "*_local_vars.json")):
                try:
                    with open(lv_path) as f:
                        lv = json.load(f)
                    local_vars_map[lv["source_id"]] = lv
                except Exception:
                    pass
        print(f"   -> Loaded {len(local_vars_map)} local_vars entries.")

        # Inject local_vars into figure evidence packets
        for d in figure_data:
            src_id = d.get("meta", {}).get("figure_id", "")
            if src_id in local_vars_map:
                d["local_vars"] = local_vars_map[src_id]

        # Inject local_vars into table evidence packets
        for d in table_data:
            src_id = d["table_name"].replace("_extracted.csv", "")
            if src_id in local_vars_map:
                d["local_vars"] = local_vars_map[src_id]

        # 5. 读取 Tab-Scheme-Seg 结果
        reactant_pool = {}
        product_pool = {}
        compound_pool = {}
        pool_path = os.path.join(intermediate_dir, "compound_pool.json")
        if os.path.exists(pool_path):
            with open(pool_path) as f:
                pool_data = json.load(f)
            # 向后兼容：若 pool_data 是旧格式 flat dict（无 reactant_pool key）
            if not isinstance(pool_data.get("reactant_pool"), dict):
                compound_pool = pool_data
                reactant_pool = {}
                product_pool = {}
            else:
                reactant_pool = pool_data.get("reactant_pool", {})
                product_pool  = pool_data.get("product_pool", {})
                compound_pool = pool_data.get("compound_pool", {})
            total = len(reactant_pool) + len(product_pool) + len(compound_pool)
            print(f"   -> Loaded compound pool: {total} entries (reactant={len(reactant_pool)}, product={len(product_pool)}, fallback={len(compound_pool)})")

        scheme_conditions = ""
        cond_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
        if os.path.exists(cond_path):
            with open(cond_path) as f:
                scheme_conditions = f.read().strip()
            print(f"   -> Loaded scheme conditions text")

        # 4. LLM Prompt
        system_prompt = (
            "You are an expert flow chemistry data extractor. "
            "Your sole task is to extract structured reaction data from flow chemistry papers. "
            "You must be precise, grounded, and never hallucinate values not present in the provided text or tables."
        )

        compound_section = ""
        if reactant_pool:
            compound_section += f"""
=== REACTANT STRUCTURE POOL (from scheme, left of reaction arrow) ===
Use for reactant_smiles when label matches:
{json.dumps(reactant_pool, indent=2)}
"""
        if product_pool:
            compound_section += f"""
=== PRODUCT STRUCTURE POOL (from scheme, right of reaction arrow) ===
Use for product_smiles when label matches:
{json.dumps(product_pool, indent=2)}
"""
        if compound_pool:
            compound_section += f"""
=== COMPOUND STRUCTURE POOL (role undetermined, no arrow detected) ===
{json.dumps(compound_pool, indent=2)}
"""

        scheme_cond_section = ""
        if scheme_conditions:
            scheme_cond_section = f"""
=== SCHEME CONDITIONS (apply to all records from this paper unless table overrides) ===
{scheme_conditions}
"""

        user_prompt = f"""You are extracting reaction data from a flow chemistry paper.

=== PAPER TEXT (truncated) ===
{full_text[:50000]}

=== EXTRACTED TABLES (CSV content) ===
{json.dumps(table_data, indent=2)}

=== EXTRACTED FIGURES (coordinate data + axis labels) ===
{json.dumps(figure_data, indent=2)}
{compound_section}{scheme_cond_section}
=== FLOW CHEMISTRY DOMAIN KNOWLEDGE ===
- For organolithium flow chemistry papers: if reactor_type is not explicitly stated in a source,
  it is typically a "T-shaped micromixer + capillary/coil reactor" setup. Apply this default ONLY
  if the paper context confirms flow microreactor use and local_vars.fixed_conditions.reactor_type is null.
- Solvent abbreviations: THF=tetrahydrofuran, Et2O=diethyl ether, DCM/CH2Cl2=dichloromethane,
  MeCN=acetonitrile, EtOAc=ethyl acetate, MeOH=methanol, toluene/PhMe, hexane, MTBE, dioxane,
  DMF=dimethylformamide, DMSO. Expand these abbreviations when filling the solvent field.
- If scheme_conditions contain solvent info, apply it to ALL records from tables in the same paper.

=== LOCAL VARIABLE LIBRARIES ===
Each table and figure above may contain a "local_vars" field. USE it to:
- Map axes/columns to the correct output fields (local_vars overrides your own interpretation).
- Apply fixed_conditions to ALL records from that source (merge with scheme conditions if not conflicting).
- Use reaction_context as context for reactant/product identification.
- Trust data_interpretation_notes for ambiguous cell or point values.

=== TASK ===
Extract ALL reaction records from the tables and figures above. Every row in a table and every data point in figure raw_data must become one record in the output — do NOT skip or merge any.
Use the paper text to fill in context (reaction conditions, abbreviation meanings, shared conditions).

For each reaction record, output one JSON object with these fields:
{{
  "reactant1_smiles": "...",       // SMILES of reactant 1 if available, else null (do NOT guess)
  "reactant1_name": "...",         // name/label of reactant 1 if SMILES not available, else null
  "reactant2_smiles": "...",       // SMILES of reactant 2 if available, else null (do NOT guess)
  "reactant2_name": "...",         // name/label of reactant 2 if SMILES not available, else null
  "product_smiles": "...",         // SMILES if available in data, else null (do NOT guess)
  "product_name": "...",           // name/label if SMILES not available, else null
  "product_label": "...",          // e.g. "4a", "compound 3", null if absent
  "yield_pct": null,               // numeric yield %, null if absent
  "yield_type": null,              // how yield was measured: "isolated" | "GC" | "NMR" | "crude" | null
  "conversion_pct": null,          // numeric conversion %, null if absent
  "selectivity_pct": null,         // numeric selectivity/regioselectivity %, null if absent
  "ee_pct": null,                  // enantiomeric excess %, null if absent
  "reaction_class": "...",          // REQUIRED — choose ONE from: "C-C coupling" | "anionic polymerization" | "carbolithiation" | "C-N coupling" | "C-O coupling" | "halogenation" | "oxidation" | "reduction" | "other". Must not be null. Use "other" if unsure. This is the same for all records from the same paper.
  "paper_doi": null,               // DOI found in paper text (e.g. "10.1039/c2cc16855c"), null if not found
  "conditions": {{
    "temperature_C": null,         // reaction temperature in °C (numeric only)
    "residence_time_s": null,      // residence time in seconds (convert ms→s if needed)
    "flow_rate_mL_min": null,      // total flow rate in mL/min
    "solvent": null,               // solvent full name(s); expand abbreviations (THF→tetrahydrofuran, DCM/CH2Cl2→dichloromethane, Et2O→diethyl ether, MeCN→acetonitrile, EtOAc→ethyl acetate, MeOH→methanol, DMF→dimethylformamide, DMSO→dimethyl sulfoxide)
    "catalyst": null,              // full catalyst/reagent text as-is from paper
    "catalyst_metal": null,        // central metal only, e.g. "Pd" | "Cu" | "Li" | "Ru" | null
    "catalyst_loading_pct": null,  // numeric mol% loading if stated, else null
    "pressure_bar": null,          // pressure in bar
    "reactor_type": null           // e.g. "microreactor", "packed bed reactor"
  }},
  "other_metrics": {{}},           // any other numeric metrics not covered above (e.g. TON, TOF, productivity g/h, K/S)
  "source_table_or_figure": "...", // prefer human-readable label e.g. "Table 1", "Figure 3"; use filename only if no label available
  "notes": null                    // any important notes
}}

=== RULES ===
1. SUBSTRATE SCOPE TABLES: Each row is one reaction record. Extract every row. If a reaction has two distinct reactants, put them in reactant1 and reactant2 fields separately — do NOT combine them into one field.
2. OPTIMIZATION/SCREENING TABLES: Extract each condition set as a separate record.
3. FIGURES: Extract EVERY SINGLE data point from figure raw_data as one separate record. Do NOT summarize or skip any points.
   (a) Standard scatter/line: axes are yield/conversion/selectivity vs a variable → each raw_data row is one record; X→condition field, Y_Left or Y_Right→metric field.
   (b) Heatmap (x_axis_title=residence time, y_axis_title=temperature): each raw_data row is one record with conditions.residence_time_s=X, conditions.temperature_C=Y_Left, yield_pct=Y_Right/Data_Value. Extract ALL rows including those with yield=0.
4. CONDITIONS: If conditions are shared across a table (stated in caption or paper text), apply them to ALL records from that table.
5. SMILES: Use SMILES only if directly provided in the paper text or figure data. If only a name is given, set reactant1_name/reactant2_name/product_name and leave SMILES null — do NOT invent SMILES.
6. SOURCE RESTRICTION: Extract data ONLY from the provided tables and figures. Do NOT extract data from the Introduction, Related Work, or References sections. If the paper discusses prior work or comparative data from other papers, skip it.
7. OTHER METRICS: If the paper reports metrics other than yield/conversion/selectivity (e.g. TON, TOF, productivity in g/h, K/S value, ee, dr, purity %), put them in "other_metrics" as a dict.
8. NO HALLUCINATION: Every numeric value must come from the extracted tables/figures. Use null for missing fields.
9. OUTPUT: Return ONLY a valid JSON array. No markdown fences, no explanation text.
10. SMILES LOOKUP:
    - If product_label matches a key in PRODUCT STRUCTURE POOL → set product_smiles to that value.
    - If reactant label matches a key in REACTANT STRUCTURE POOL → set reactant1_smiles or reactant2_smiles accordingly.
    - If only COMPOUND STRUCTURE POOL exists (no arrow detected), use context to infer role and assign accordingly.
    Do not modify SMILES strings.
11. SCHEME CONDITIONS: If Scheme Conditions are provided and a table record lacks certain condition fields (temperature, solvent, catalyst), use the Scheme Conditions as fallback.
12. LOCAL VARS: If a source has a "local_vars" field, its axis_semantics and fixed_conditions OVERRIDE your general interpretation. Trust local_vars.data_interpretation_notes for ambiguous cell or point values.
13. REACTION CLASS: reaction_class MUST be filled for every record. Read the paper text, identify the main reaction type, and pick the best match from the allowed list. Use "other" if none fit. Never leave reaction_class as null.

Output a JSON array of all extracted reaction records:"""
        
        print("   -> Sending to LLM...")
        response = self.llm.chat(system_prompt, user_prompt)

        # Strip markdown fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip().rsplit("\n", 1)[0]

        # Remove invalid control characters (except tab/newline/CR) that break JSON parsing
        import re as _re
        cleaned = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

        # 5. Save Output
        with open(out_file, 'w') as f:
            f.write(cleaned)

        print(f"   -> Saved final result to {out_file}")

        # 6. Excel 输出
        try:
            records = json.loads(cleaned)
            if isinstance(records, list):
                self._save_excel(records, basename)
        except Exception as e:
            print(f"[GlobalAssembly] Excel export skipped: {e}")

        return out_file

    def _save_excel(self, records, basename):
        import pandas as pd
        out_path = os.path.join(self.output_dir, f"{basename}_final.xlsx")
        flat = []
        for r in records:
            row = dict(r)
            conds = row.pop("conditions", {}) or {}
            other = row.pop("other_metrics", {}) or {}
            row.update(conds)
            row.update(other)
            flat.append(row)
        df = pd.DataFrame(flat)
        src_col = "source_table_or_figure"
        if src_col not in df.columns:
            df[src_col] = "unknown"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for src, grp in df.groupby(src_col, sort=False):
                sheet = str(src)[:31]
                grp.to_excel(writer, sheet_name=sheet, index=False)
        print(f"   -> Saved Excel to {out_path}")
