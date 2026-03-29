import os
import json
import glob
import sys
from src.adjudication.llm_engine import LLMEngine, sanitize_json_text
from src.adjudication.pdf_parser import PDFParser

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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
                with open(out_file, encoding="utf-8") as f:
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
                        with open(jpath, 'r', encoding="utf-8") as f:
                            d = json.load(f)
                        meta = d.get('meta', {})
                        source_dir = meta.get('source_intermediate_dir', '')
                        if basename in source_dir or evidence_dir == macro_cleaned_dir:
                            figure_data.append(d)
                    except Exception as _e:
                        print(f"[GlobalAssembly] Warning: skipped evidence file {jpath}: {_e}")

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
                         with open(csv_path, 'r', encoding="utf-8", errors="replace") as f:
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
                    lv = self._load_json_with_fallback(lv_path)
                    if lv is None:
                        raise ValueError("unreadable local_vars")
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
            with open(pool_path, encoding="utf-8") as f:
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
            with open(cond_path, encoding="utf-8") as f:
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
  "entry_number": null,            // table entry number, e.g. "1", "2a", null if absent
  "yield_pct": null,               // numeric yield %, null if absent
  "yield_type": null,              // how yield was measured: "isolated" | "GC" | "NMR" | "crude" | null
  "batch_yield_pct": null,         // yield of the analogous batch reaction for comparison, numeric %, null if absent; e.g. if paper says "batch yield 34%", set 34.0
  "conversion_pct": null,          // numeric conversion %, null if absent
  "selectivity_pct": null,         // numeric selectivity/regioselectivity %, null if absent
  "ee_pct": null,                  // enantiomeric excess %, null if absent
  "diastereomeric_ratio": null,    // diastereoselectivity as string, e.g. "anti:syn = 99:1", null if absent
  "stoichiometry": null,           // molar ratio or equivalents, e.g. "[M]/[RAFT]=100", "1.2 equiv", null if absent
  "reaction_class": "...",         // REQUIRED — choose ONE from: "nucleophilic addition" | "halogen-metal exchange" | "directed metalation" | "anionic cyclization" | "C-C coupling" | "C-N coupling" | "C-O coupling" | "polymerization" | "halogenation" | "oxidation" | "reduction" | "hydrogenation" | "esterification" | "amidation" | "alkylation" | "acylation" | "photocatalysis" | "hydrolysis" | "other". Must not be null. Use "other" if unsure. For organolithium papers: prefer "nucleophilic addition" (RLi + carbonyl/imine), "halogen-metal exchange" (ArX + RLi → ArLi), "directed metalation" (C-H deprotonation), "anionic cyclization" (intramolecular). This is the same for all records from the same paper.
  "paper_doi": null,               // DOI found in paper text (e.g. "10.1039/c2cc16855c"), null if not found
  "conditions": {{
    "temperature_C": null,         // reaction temperature in °C (numeric only)
    "residence_time_s": null,      // residence time in seconds (convert ms→s if needed)
    "flow_rate_mL_min": null,      // total combined flow rate in mL/min
    "flow_rate_stream1_mL_min": null, // flow rate of organic phase / first stream (mL/min), null if not separately stated; e.g. Qorg=0.5 mL/min → 0.5
    "flow_rate_stream2_mL_min": null, // flow rate of aqueous phase / second stream (mL/min), null if not separately stated; e.g. Qaq=0.3 mL/min → 0.3
    "solvent": null,               // solvent full name(s); expand abbreviations (THF→tetrahydrofuran, DCM/CH2Cl2→dichloromethane, Et2O→diethyl ether, MeCN→acetonitrile, EtOAc→ethyl acetate, MeOH→methanol, DMF→dimethylformamide, DMSO→dimethyl sulfoxide)
    "catalyst": null,              // catalyst main body name ONLY — no ligand, no loading; e.g. "CuBr·Me2S" not "CuBr·Me2S (5 mol%)"
    "catalyst_metal": null,        // central metal only, e.g. "Pd" | "Cu" | "Li" | "Ru" | null
    "catalyst_loading_pct": null,  // numeric mol% loading extracted from catalyst string, e.g. "Pd(OAc)2 (2 mol%)" → 2.0; null if not stated
    "ligand": null,                // ligand name only (split from catalyst string), e.g. "BINAP", "L2", null if absent
    "ligand_loading_pct": null,    // ligand loading in mol%, e.g. "L2 (6 mol%)" → 6.0, null if absent
    "additive": null,              // additive or co-reagent not counted as catalyst or ligand, e.g. "BF3·OEt2", "K2CO3", null if absent
    "pressure_bar": null,          // pressure in bar
    "reactor_type": null           // e.g. "microreactor", "packed bed reactor"
  }},
  "other_metrics": {{}},           // any other numeric metrics not covered above (e.g. TON, TOF, productivity g/h, K/S)
  "source_table_or_figure": "...", // prefer human-readable label e.g. "Table 1", "Figure 3"; use filename only if no label available
  "data_correction_note": null,    // ONLY OCR correction or numeric inference note, max 1 sentence; e.g. "OCR read '8l' corrected to '81'"; null if no correction needed
  "notes": null                    // ONLY narrative context that cannot fit any structured field above; if all info is captured elsewhere, set null
}}

=== RULES ===
1. SUBSTRATE SCOPE TABLES: Each row is one reaction record. Extract every row. If a reaction has two distinct reactants, put them in reactant1 and reactant2 fields separately — do NOT combine them into one field.
2. OPTIMIZATION/SCREENING TABLES: Extract each condition set as a separate record.
3. FIGURES: Extract EVERY SINGLE data point from figure raw_data as one separate record. Do NOT summarize, merge, or skip any points.
   IMPORTANT COUNT CHECK: if a figure evidence packet has N items in raw_data, you MUST output exactly N records from that figure.
   (a) Standard scatter/line: each raw_data row is one record.
       - X value → place into the condition field named by local_vars.axis_semantics.x_axis.maps_to_field (e.g. if maps_to_field="conditions.flow_rate_mL_min", set conditions.flow_rate_mL_min=X). NEVER leave X unused.
       - Y_Left → the metric field named by local_vars.axis_semantics.y_left_axis.maps_to_field (e.g. selectivity_pct=Y_Left).
       - Y_Right/Data_Value → the metric field named by local_vars.axis_semantics.y_right_axis.maps_to_field (if present).
       - Series name → product_name (or reactant_name if the series represents a reactant).
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
    - If a reactant/product name is only a generic label like "Compound 1", first resolve it to the full chemical name from the scheme, caption, paper text, or local_vars.reaction_context. Only keep the numbered label if no explicit name is available.
11. SCHEME CONDITIONS: If Scheme Conditions are provided and a table record lacks certain condition fields (temperature, solvent, catalyst), use the Scheme Conditions as fallback.
12. LOCAL VARS: If a source has a "local_vars" field:
    - axis_semantics.x_axis.maps_to_field tells you which output field to assign the X value to (MANDATORY — never leave X unused).
    - axis_semantics.y_left_axis.maps_to_field and y_right_axis.maps_to_field tell you which output fields to assign Y_Left and Y_Right/Data_Value to.
    - fixed_conditions apply to ALL records from that source.
    - data_interpretation_notes explain how to interpret ambiguous values.
    These OVERRIDE your own interpretation.
13. REACTION CLASS: reaction_class MUST be filled for every record. Read the paper text, identify the main reaction type, and pick the best match from the allowed list. Use "other" if none fit. Never leave reaction_class as null.
14. NOTES: Only use the notes field for narrative context that cannot fit any other field. If all information is captured in other fields, set notes to null. Do NOT repeat information already in other fields such as entry_number, diastereomeric_ratio, batch_yield_pct, stoichiometry, flow rate streams, ligand, or data_correction_note.
15. CATALYST SPLITTING: Write only the catalyst main body in "catalyst" (e.g. "CuI", "Pd(OAc)2"). Extract the loading into catalyst_loading_pct. Put the ligand name in "ligand" and its loading in "ligand_loading_pct". Put any remaining co-reagent (not the main catalyst, not a ligand) in "additive".

Output a JSON array of all extracted reaction records:"""
        
        print("   -> Sending to LLM...")
        response = self.llm.chat(system_prompt, user_prompt)

        # Save raw LLM response for debugging (overwritten each run)
        raw_path = os.path.join(self.output_dir, f"{basename}_final_raw.txt")
        try:
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(response)
        except Exception:
            pass

        cleaned = sanitize_json_text(response)

        # 5. Parse JSON – save [] on failure so PostProcessor gets valid (empty) input
        try:
            records = json.loads(cleaned)
            if isinstance(records, dict):
                for key in ("records", "items", "data", "output", "result"):
                    val = records.get(key)
                    if isinstance(val, list):
                        records = val
                        break
                else:
                    if records and all(not isinstance(v, (dict, list)) for v in records.values()):
                        records = [records]
            if not isinstance(records, list):
                raise ValueError(f"Expected JSON array, got {type(records).__name__}")
        except Exception as e:
            recovered = self._recover_json_records(response)
            if recovered:
                print(f"[GlobalAssembly] JSON parse recovered {len(recovered)} records from malformed array.")
                records = recovered
                cleaned = json.dumps(records, ensure_ascii=False, indent=2)
            else:
                print(f"[GlobalAssembly] JSON parse failed: {e}  (raw saved to {raw_path})")
                records = []
                cleaned = "[]"

        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"   -> Saved final result to {out_file} ({len(records)} records)")

        # 6. Excel 输出
        if records:
            try:
                self._save_excel(records, basename)
            except Exception as e:
                print(f"[GlobalAssembly] Excel export skipped: {e}")

        return out_file

    def _load_json_with_fallback(self, path):
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "gbk"):
            try:
                with open(path, encoding=encoding) as f:
                    return json.load(f)
            except Exception:
                continue
        return None

    def _recover_json_records(self, text):
        """
        Best-effort recovery for malformed JSON arrays returned by the LLM.
        Scans for top-level object spans and parses each object independently.
        """
        if not text:
            return []
        start = text.find("[")
        if start == -1:
            return []
        payload = text[start + 1:]
        records = []
        depth = 0
        in_string = False
        escape = False
        obj_start = None
        for idx, ch in enumerate(payload):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    obj_start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        chunk = payload[obj_start:idx + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                records.append(obj)
                        except Exception:
                            pass
                        obj_start = None
        return records

    def _save_excel(self, records, basename):
        import pandas as pd
        from src.adjudication.post_processor import PREFERRED_COLUMNS
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

        # Apply fixed column order
        ordered = [c for c in PREFERRED_COLUMNS if c in df.columns]
        extra = sorted(c for c in df.columns if c not in PREFERRED_COLUMNS)
        df = df[ordered + extra]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Records", index=False)
            for src, grp in df.groupby(src_col, sort=False):
                sheet = str(src)[:31]
                grp.to_excel(writer, sheet_name=sheet, index=False)
        print(f"   -> Saved Excel to {out_path}")
