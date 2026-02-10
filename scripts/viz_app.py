import streamlit as st
import os
import sys

# --- GLOBAL RESOURCE LIMITS (Must be set before importing torrent/cv2/etc in subprocesses) ---
# Prevent overheating and zombie processes by forcing single-threaded execution for libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true" 

import glob
import subprocess
import json
import pandas as pd
import time
from PIL import Image
import cv2

# Force OpenCV single thread
cv2.setNumThreads(0)

st.set_page_config(layout="wide", page_title="Unified FlowFigTabMiner Dashboard")

# --- Constants & Utils ---
PYTHON_EXEC = sys.executable if sys.executable else "python3"
# Or fallback to relative path if running in venv
if os.path.exists("./flowfigtabminer/bin/python3"):
    PYTHON_EXEC = "./flowfigtabminer/bin/python3"

def run_script(script_path, args=[]):
    """Run a script and capture output."""
    cmd = [PYTHON_EXEC, script_path] + args
    # CRITICAL: Pass current environment (with thread limits) to subprocess
    # subprocess.run inherits env by default, but let's be explicit if needed.
    # Actually, default behavior IS to inherit os.environ.
    # But let's verify if we need to force it or if there's a shell=True issue (we are not using shell=True).
    
    # Debug: Print command being run
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        return result
    except Exception as e:
        print(f"Subprocess failed: {e}")
        # Return a dummy completed process with error
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, 1, stdout="", stderr=str(e))

def parse_json_from_stdout(stdout):
    """Extract JSON from ---JSON_START--- ... ---JSON_END--- block or last valid JSON line."""
    try:
        if "---JSON_START---" in stdout:
            json_str = stdout.split("---JSON_START---")[1].split("---JSON_END---")[0]
            return json.loads(json_str)
    except:
        pass
    return None

# --- UI Layout ---
st.title("🧪 FlowFigTabMiner Unified Dashboard")

# Sidebar: Document Selection
st.sidebar.header("Document Selection")
input_dir = "data/input"
if not os.path.exists(input_dir):
    st.sidebar.error(f"Input directory not found: {input_dir}")
    st.stop()

pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
pdf_map = {os.path.basename(f): f for f in pdf_files}

selected_pdf_name = st.sidebar.selectbox("Select PDF", list(pdf_map.keys()))

if not selected_pdf_name:
    st.info("Please select a PDF to start.")
    st.stop()

selected_pdf_path = pdf_map[selected_pdf_name]
pdf_basename = os.path.splitext(selected_pdf_name)[0]
intermediate_dir = os.path.join("data/intermediate", pdf_basename)
figures_dir = os.path.join(intermediate_dir, "figures")
tables_dir = os.path.join(intermediate_dir, "tables") # Convention: TF-ID puts raw table tables here? 
# Actually TF-ID (Step 1) usually puts 'figures' and 'tables' in intermediate/{basename}/...
# Let's verify standard paths later.

# --- Pipeline Status Indicators ---
st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Status")

has_step1 = os.path.exists(figures_dir)
st.sidebar.checkbox("Step 1 (TF-ID)", value=has_step1, disabled=True)

# Step 2-4 Figures?
# Check if evidence exists?
evidence_dir = "data/evidence" # This is global?
# We need to know if figures for THIS pdf are done.
# Simple check: `data/evidence/*_evidence.json` where ID matches?
has_figures = False # heuristic

# Step Tables?
has_tables = False # heuristic

# Step 5?
final_json = os.path.join("data/final_output", f"{pdf_basename}_final.json")
has_step5 = os.path.exists(final_json)
st.sidebar.checkbox("Step 5 (Assembly)", value=has_step5, disabled=True)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run Full Pipeline", type="primary"):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # 1. Step 1: TF-ID
        status_text.write("Step 1/4: Running TF-ID Extraction...")
        res1 = run_script("scripts/step1_tfid.py", [selected_pdf_path])
        if res1.returncode != 0:
            st.sidebar.error("Step 1 Failed!")
            st.error(res1.stderr)
            st.stop()
        progress_bar.progress(25)
        
        # 2. Step 2-4: Figures
        status_text.write("Step 2/4: Processing Figures...")
        res2 = run_script("scripts/run_steps2_to_4.py", [selected_pdf_path])
        if res2.returncode != 0:
            st.sidebar.warning("Figure pipeline had issues (check logs), continuing...")
            st.write("Figure Logs:", res2.stderr)
        progress_bar.progress(50)
        
        # 3. Step Tables
        status_text.write("Step 3/4: Processing Tables...")
        # Target specific table directory for this PDF
        pdf_tables_dir = os.path.join(intermediate_dir, "tables")
        if os.path.exists(pdf_tables_dir):
            # We run batch pipeline on this specific folder
            res3 = run_script("scripts/run_batch_tables.py", ["--input_dir", pdf_tables_dir])
            if res3.returncode != 0:
                 st.sidebar.warning("Table pipeline had issues, continuing...")
                 st.write("Table Logs:", res3.stderr)
        else:
            st.sidebar.info("No tables found to process.")
        progress_bar.progress(75)

        # 4. Step 5: Assembly
        status_text.write("Step 4/4: Final Assembly & Synthesis...")
        res5 = run_script("scripts/step5_advanced.py", [selected_pdf_path])
        if res5.returncode != 0:
             st.sidebar.error("Step 5 Failed!")
             st.error(res5.stderr)
             st.stop()
        progress_bar.progress(100)
        
        status_text.write("✅ Pipeline Complete!")
        st.sidebar.success("All Steps Finished.")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"Pipeline Error: {e}")

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Preparation (TF-ID)", 
    "2. Figure Pipeline", 
    "3. Table Pipeline", 
    "4. Global Assembly"
])

# === TAB 1: PREPARATION ===
with tab1:
    st.header("Step 1: Raw Extraction (TF-ID)")
    
    col_act, col_info = st.columns([1, 2])
    with col_act:
        if st.button("Run Step 1 (TF-ID)", type="primary"):
            with st.spinner("Running TF-ID extraction..."):
                res = run_script("scripts/step1_tfid.py", [selected_pdf_path])
                if res.returncode == 0:
                    st.success("Step 1 Complete!")
                    st.rerun()
                else:
                    st.error("Step 1 Failed")
                    with st.expander("Logs"):
                        st.text(res.stderr)
                        st.text(res.stdout)

    # Visualization of Assets
    if os.path.exists(intermediate_dir):
        # Figures
        fig_assets = sorted(glob.glob(os.path.join(figures_dir, "*.png")))
        # Tables
        # TF-ID output structure might vary. Assume intermediate/{basename}/tables/*.png
        # If TF-ID outputting tables to 'tables' subfolder
        tab_assets = sorted(glob.glob(os.path.join(intermediate_dir, "tables", "*.png")))
        # Filter out processed/debug
        tab_assets_raw = [f for f in tab_assets if "_body" not in f and "_crop" not in f and "_smiles" not in f]

        st.subheader(f"Extracted Assets ({len(fig_assets)} Figures, {len(tab_assets_raw)} Tables)")
        
        with st.expander("View Raw Figures", expanded=False):
            if fig_assets:
                cols = st.columns(4)
                for i, p in enumerate(fig_assets):
                    with cols[i % 4]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
            else:
                st.info("No figures found.")

        with st.expander("View Raw Tables", expanded=False):
            if tab_assets_raw:
                cols = st.columns(3)
                for i, p in enumerate(tab_assets_raw):
                    with cols[i % 3]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
            else:
                st.info("No tables found.")
    else:
        st.info("Run Step 1 to generate intermediate data.")


# === TAB 2: FIGURE PIPELINE ===
with tab2:
    st.header("Step 2-4: Figure Data Mining")
    
    if st.button("Run Figure Pipeline (All Figures)", type="primary"):
        with st.spinner("Running Logic: Macro -> Micro -> Legend -> Assembly..."):
            res = run_script("scripts/run_steps2_to_4.py", [selected_pdf_path])
            
            with st.expander("Execution Logs", expanded=True):
                st.text(res.stdout)
                if res.stderr:
                    st.text(res.stderr)
            
            if res.returncode == 0:
                st.success("Figure Pipeline Complete!")

    # --- Debug Mode: Single Figure Step-by-Step ---
    with st.expander("🛠 Advanced Debug: Single Figure Steps"):
        st.markdown("Run individual steps on a specific figure crop to debug issues.")
        fig_assets = sorted(glob.glob(os.path.join(figures_dir, "*.png")))
        if fig_assets:
            sel_fig_debug = st.selectbox("Select Raw Figure Crop", [os.path.basename(f) for f in fig_assets])
            sel_fig_path = os.path.join(figures_dir, sel_fig_debug)
            
            col_dbg_1, col_dbg_2 = st.columns(2)
            
            with col_dbg_1:
                st.image(sel_fig_path, caption="Raw Input", width=300)
                
                # Define unique keys for this debug session
                s2_key = f"s2_{sel_fig_debug}"
                s3_key = f"s3_{sel_fig_debug}"

                # Step 2
                if st.button("Run Step 2 (Macro Clean)", key="btn_step2"):
                    res = run_script("scripts/step2_macro_single.py", [sel_fig_path])
                    
                    # Show logs
                    with st.expander("Step 2 Logs (Stdout/Stderr)", expanded=False):
                         st.text(res.stdout)
                         if res.stderr:
                             st.text(f"STDERR:\n{res.stderr}")

                    if res.returncode == 0:
                        st.success("Step 2 Done")
                        j = parse_json_from_stdout(res.stdout)
                        if j:
                            st.session_state[s2_key] = j
                            # Clear subsequent steps if Step 2 re-run
                            if s3_key in st.session_state: del st.session_state[s3_key]
                    else:
                        st.error("Step 2 Failed")
                        if res.stderr: st.error(res.stderr)
                
                # Render Step 2 Results (Persistent)
                if s2_key in st.session_state:
                    j = st.session_state[s2_key]
                    
                    # Show cleaned image
                    cleaned_rel = j.get("cleaned_image")
                    if cleaned_rel and os.path.exists(cleaned_rel):
                        st.image(cleaned_rel, caption="Macro Cleaned Result", width=300)
                    
                    # Show elements
                    elems = j.get("elements", {})
                    if elems:
                         st.markdown("**Detected Elements (Masked):**")
                         ec = st.columns(len(elems))
                         for idx, (label, paths) in enumerate(elems.items()):
                             with ec[idx]:
                                 st.caption(label)
                                 for p in paths:
                                     if os.path.exists(p):
                                         img_data = p
                                         width_val = 150
                                         if label == "y_axis_title":
                                             try:
                                                 pil_img = Image.open(p)
                                                 img_data = pil_img.rotate(-90, expand=True)
                                                 width_val = 100
                                             except: pass
                                         elif label in ["legend", "x_axis_title"]:
                                             width_val = 300
                                         st.image(img_data, width=width_val)

            with col_dbg_2:
                # Step 3
                # Need clean image locally or from state?
                # We can deduce path from persisting state or glob (as before)
                # But glob is safer if user didn't just run Step 2 but file exists.
                # Let's stick to glob logic but allow Step 3 results persistence.
                
                base_name = os.path.splitext(sel_fig_debug)[0]
                macro_dir = os.path.join(intermediate_dir, "macro_cleaned")
                candidates = glob.glob(os.path.join(macro_dir, f"{base_name}_t*_cleaned.png"))
                
                if candidates:
                    sel_clean = candidates[0]
                    if len(candidates) > 1:
                        sel_clean = st.selectbox("Select Cleaned Crop", [os.path.basename(c) for c in candidates])
                        sel_clean = os.path.join(macro_dir, sel_clean)
                    
                    st.image(sel_clean, caption="Cleaned Input for Step 3", width=300)
                    
                    if st.button("Run Step 3 (Micro Detect)", key="btn_step3"):
                        res = run_script("scripts/step3_micro_single.py", [sel_clean])
                        
                        with st.expander("Step 3 Logs", expanded=False):
                            st.text(res.stdout)
                            if res.stderr: st.text(f"STDERR:\n{res.stderr}")

                        if res.returncode == 0:
                             st.success("Step 3 Done")
                             j = parse_json_from_stdout(res.stdout)
                             if j: st.session_state[s3_key] = j
                    
                    # Render Step 3 Results (Persistent)
                    if s3_key in st.session_state:
                         j = st.session_state[s3_key]
                         
                         # Data
                         # 1. Data
                         mapped = j.get("mapped_data", [])
                         if mapped:
                             st.markdown(f"### Extracted Data ({len(mapped)} pts)")
                             st.dataframe(pd.DataFrame(mapped))
                         else:
                             st.warning("No data mapped.")
                                 
                         # 2. Key Stats
                         st.markdown(f"**Stats:** Points: {j.get('num_points_detected', 0)} | Legends: {j.get('num_legends_found', 0)}")
                         
                         # 3. Visualization of Detections
                         detections = j.get("detections", [])
                         if detections and os.path.exists(sel_clean):
                             import cv2
                             import numpy as np
                             
                             # Load image to draw on
                             img_vis = cv2.imread(sel_clean)
                             img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
                             
                             for d in detections:
                                 bbox = d.get('box')
                                 label = d.get('label')
                                 conf = d.get('conf', 0.0)
                                 
                                 if bbox:
                                     x1, y1, x2, y2 = map(int, bbox)
                                     # Color code
                                     color = (255, 0, 0) # Red default
                                     if label == 'data_point': color = (0, 255, 0) # Green
                                     elif label == 'x_tick_label': color = (0, 0, 255) # Blue
                                     elif label == 'y_tick_label': color = (255, 0, 255) # Magenta
                                     elif 'tick_label' in label: color = (0, 0, 255) # Fallback Blue
                                     elif label == 'tick_mark': color = (255, 255, 0) # Yellow
                                     
                                     cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                                     cv2.putText(img_vis, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                             
                             st.image(img_vis, caption="YOLO Micro Detections", width=400)

                         # 4. Debug Log
                         with st.expander("Detailed Logic Log"):
                             for l in j.get("debug_log", []):
                                 st.text(l)

                         # --- NEW: Step 4 Assembly ---
                         st.markdown("---")
                         st.markdown("**Step 4: Assembly**")
                         if st.button("Run Step 4 (Assemble Evidence)", key="btn_step4"):
                             # 1. Save Step 3 Data to Temp File
                             temp_s3_path = os.path.join(intermediate_dir, f"temp_step3_{base_name}.json")
                             with open(temp_s3_path, 'w') as f:
                                 json.dump(j, f)
                             
                             # 2. Run Script
                             # Usage: step4_assembly_single.py <fig_id> <macro_dir> <step3_json>
                             # fig_id is base_name (removed _cleaned)
                             fig_id_clean = base_name.replace("_cleaned", "")
                             
                             res = run_script("scripts/step4_assembly_single.py", [
                                 fig_id_clean, 
                                 macro_dir, 
                                 temp_s3_path
                             ])
                             
                             with st.expander("Step 4 Logs", expanded=True):
                                 st.text(res.stdout)
                                 if res.stderr: st.text(f"STDERR:\n{res.stderr}")
                             
                             if res.returncode == 0:
                                 s4_json = parse_json_from_stdout(res.stdout)
                                 if s4_json and s4_json.get("status") == "success":
                                     st.success(f"Evidence Assembled! Saved to: {s4_json['output_path']}")
                                     st.json(s4_json.get("evidence"))
                                 else:
                                     st.warning("Assembly finished but filtered out (or no JSON returned). Check logs.")
                             else:
                                 st.error("Step 4 Failed")

                else:
                    st.info("Run Step 2 first to generate cleaned image.")
        else:
            st.info("No figures found to debug.")

    # Visualization of Results (Moved to Bottom)
    st.markdown("---")
    st.subheader("Results Gallery")
    
    ev_files = glob.glob("data/evidence/*.json")
    relevant_ev = []
    for f in ev_files:
        try:
            with open(f, 'r') as jf:
                d = json.load(jf)
                if pdf_basename in d.get('meta', {}).get('source_intermediate_dir', ''):
                    relevant_ev.append(f)
        except: pass
    
    if relevant_ev:
        sel_ev = st.selectbox("Select Evidence Packet", [os.path.basename(f) for f in relevant_ev])
        if sel_ev:
            full_path = os.path.join("data/evidence", sel_ev)
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Metadata**")
                st.json(data['meta'])
                
                fid = data['meta']['figure_id']
                macro_dir = os.path.join(intermediate_dir, "macro_cleaned")
                img_path = os.path.join(macro_dir, f"{fid}.png")
                if not os.path.exists(img_path):
                     img_path = os.path.join(macro_dir, f"{fid}_cleaned.png")
                
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Analyzed Image: {fid}", use_container_width=True)

            with c2:
                st.markdown("**Extracted Data Points**")
                raw = data.get('raw_data', [])
                if raw:
                    st.dataframe(pd.DataFrame(raw))
                else:
                    st.warning("No data points.")
                
                st.markdown("**Text Evidence**")
                st.json(data.get('text_evidence', {}))

    else:
        st.info("No evidence found for this PDF. Run pipeline above.")


# === TAB 3: TABLE PIPELINE ===
with tab3:
    st.header("Step Table: Structure & Molecule Mining")
    
    # Find raw tables
    raw_tables = sorted(glob.glob(os.path.join(intermediate_dir, "tables", "*.png")))
    # Filter out sub-images (body, smiles, etc) to find ROOTS
    # Root tables from TF-ID are usually just {pdf}_table_{i}.png
    # But processed ones create subfolders?
    # Or just files in same dir? 
    # Let's filter strictly.
    raw_tables = [f for f in raw_tables if "_body" not in f and "_smiles" not in f and "_crop" not in f and "_viz" not in f]
    
    if not raw_tables:
        st.info("No tables found via TF-ID.")
    else:
        # Selection
        t_names = [os.path.basename(f) for f in raw_tables]
        sel_t = st.selectbox("Select Table Image", t_names)
        sel_t_path = os.path.join(intermediate_dir, "tables", sel_t)
        
        st.image(sel_t_path, caption="Original Table", width=400)
        
        t_base = os.path.splitext(sel_t)[0]
        # We use a subfolder for outputs to keep things clean
        t_out = os.path.join(intermediate_dir, "tables", t_base)
        os.makedirs(t_out, exist_ok=True)

        # Actions
        st.subheader("Pipeline Actions")
        
        # Unified Button
        if st.button("Run Full Pipeline (All Steps)"):
            with st.spinner("Processing..."):
                res = run_script("scripts/step_table_pipeline.py", [sel_t_path, "--output_dir", t_out])
                with st.expander("Full Pipeline Logs"):
                    st.text(res.stdout)
                    st.text(res.stderr)
                if res.returncode == 0:
                    st.success("Full Pipeline Complete!")
                    st.session_state['last_table_run'] = t_base

        st.markdown("---")
        st.markdown("**Step-by-Step Debugging**")
        
        c_step2, c_step3, c_step4 = st.columns(3)
        
        # Step 2: Segmentation
        with c_step2:
            st.markdown("**1. Segmentation (YOLO)**")
            if st.button("Run Step 1 Start", key="btn_t_s1"):
                res = run_script("scripts/step_table_segmentation.py", [sel_t_path, "--output_dir", t_out])
                if res.returncode == 0:
                    st.success("Done")
                    # Debug Logs
                    with st.expander("Step 1 Execution Logs"):
                        st.text(res.stdout)
                        st.text(res.stderr)
                    j = parse_json_from_stdout(res.stdout)
                    if j: st.session_state[f'seg_json_{t_base}'] = j
                else:
                    st.error("Failed")
                    st.text(res.stdout)
                    st.text(res.stderr)
            
            # Show Results (Persistent)
            if f'seg_json_{t_base}' in st.session_state:
                j = st.session_state[f'seg_json_{t_base}']
                
                # Logs
                with st.expander("Segmentation Logs", expanded=False):
                    logs = j.get('logs', [])
                    if logs:
                        st.dataframe(pd.DataFrame(logs)[['label','confidence','saved_path']])
                    else:
                        st.text("No components found.")
                
                # Images
                if j.get('best_body_crop_path'):
                   st.image(j['best_body_crop_path'], caption="Best Body Crop", use_container_width=True)
                
                # Component Gallery
                st.caption("Other Components:")
                logs = j.get('logs', [])
                if logs:
                     # Filter out the body if it's best? No just show all unique
                     for l in logs:
                         p = l.get('saved_path')
                         if p and os.path.exists(p) and "body_main" not in p:
                             st.image(p, caption=f"{l['label']} ({l['confidence']:.2f})", width=150)

        # Step 3: Structure (+Molecule)
        # Input is body crop (prefer the "Best Body" from Step 2 if avail, else default)
        # We try to find the best body path from session state or guess
        best_body = os.path.join(t_out, f"{t_base}_body_main.png")
        if not os.path.exists(best_body):
             best_body = os.path.join(t_out, f"{t_base}_body.png") # Old default name
        
        with c_step3:
            st.markdown("**2. Structure & Molecule**")
            if os.path.exists(best_body):
                st.image(best_body, caption="Input Body", width=100)
                
                if st.button("Run Step 2 Start", key="btn_t_s2"):
                     res = run_script("scripts/step_table_structure.py", [best_body])
                     if res.returncode == 0:
                         st.success("Done")
                         # Debug Logs
                         with st.expander("Step 2 Execution Logs", expanded=True):
                             st.text(res.stdout)
                             st.text(res.stderr)
                         j = parse_json_from_stdout(res.stdout)
                         if j: st.session_state[f'struct_json_{t_base}'] = j
                     else:
                         st.error("Failed")
                         st.text(res.stdout)
                         st.text(res.stderr)
            else:
                st.warning("No Body Crop found. Run Step 1.")

            # Show Results
            if f'struct_json_{t_base}' in st.session_state:
                j = st.session_state[f'struct_json_{t_base}']
                
                # 1. Viz Image
                viz = j.get('viz_path')
                if viz and os.path.exists(viz):
                    st.image(viz, caption="Structure Prediction", use_container_width=True)
                
                # 2. Molecule SMILES Image?
                # The script implicitly overwrites if molecule found? 
                # Or checks if _smiles.png exists in output dir
                smiles_path = best_body.replace(".png", "_smiles.png")
                if os.path.exists(smiles_path):
                     st.image(smiles_path, caption="Molecule Replaced", use_container_width=True)
                
                # 3. Logs
                with st.expander("Structure Logs"):
                     logs = j.get('logs', [])
                     if logs:
                         df_log = pd.DataFrame(logs)
                         # Clean up box column
                         st.dataframe(df_log)

        # Step 4: OCR Assembly
        struct_json_path = os.path.join(t_out, f"{t_base}_structure.json")
        
        with c_step4:
            st.markdown("**3. OCR Assembly**")
            
            # We need the structure JSON file on disk
            # If we have it in session, we can save it to be safe
            if f'struct_json_{t_base}' in st.session_state:
                 # Check if need to write
                 # Always write to sync state
                 with open(struct_json_path, 'w') as f:
                     json.dump(st.session_state[f'struct_json_{t_base}'], f)
            
            if os.path.exists(best_body) and os.path.exists(struct_json_path):
                if st.button("Run Step 3 Start", key="btn_t_s3"):
                    res = run_script("scripts/step_table_assembly.py", [
                        best_body, 
                        struct_json_path, 
                        "--output_dir", t_out,
                        "--padding", "14"
                    ])
                    
                    if res.returncode == 0:
                        st.success("Done")
                        j = parse_json_from_stdout(res.stdout)
                        if j: st.session_state[f'assembly_json_{t_base}'] = j
                    else:
                        st.error("Failed")
                        st.text(res.stderr)
            else:
                if not os.path.exists(best_body): st.warning("Need Body Crop")
                if not os.path.exists(struct_json_path): st.warning("Need Structure JSON (Run Step 2)")
            
            # Show Results
            if f'assembly_json_{t_base}' in st.session_state:
                j = st.session_state[f'assembly_json_{t_base}']
                
                csv_path = j.get('csv_path')
                json_path = j.get('json_path')
                
                if json_path and os.path.exists(json_path):
                     st.success(f"Evidence Saved: `{os.path.basename(json_path)}`")
                     with st.expander("Show Full Evidence JSON", expanded=True):
                         with open(json_path, 'r') as f:
                             st.json(json.load(f))
                
                if csv_path and os.path.exists(csv_path):
                     st.write(f"CSV: `{os.path.basename(csv_path)}`")
                     st.dataframe(pd.read_csv(csv_path, header=None))
                
                with st.expander("Cell OCR Logs"):
                    cell_logs = j.get('cell_logs', [])
                    if cell_logs:
                        st.dataframe(pd.DataFrame(cell_logs)[['row', 'col', 'text']])

# === TAB 4: GLOBAL ASSEMBLY ===
with tab4:
    st.header("Step 5: Global Assembly (LLM)")
    
    st.markdown("Aggregates all extracted Figures and Tables, truncates PDF text, and sends to LLM.")
    
    if st.button("Run Global Assembly (Step 5)", type="primary"):
        with st.status("Running LLM Global Assembly (Advanced Context-Aware)...", expanded=True) as status:
            t_start = time.time()
            st.write("Generating structured data with Context-Aware Logic...")
            
            # Run new advanced script
            # scripts/step5_advanced.py {pdf_path}
            cmd_args = [selected_pdf_path]
            
            # --- EXECUTION ---
            ret = run_script("scripts/step5_advanced.py", cmd_args)
            
            st.write(f"Done in {time.time() - t_start:.2f}s")
            
            # --- LOGS ---
            with st.expander("Step 5 Execution Logs", expanded=False):
                st.code(ret.stdout, language="text")
                if ret.stderr:
                    st.error("Stderr Output:")
                    st.code(ret.stderr, language="text")
            
            # Parse Result
            json_start = ret.stdout.find("---JSON_START---")
            if json_start != -1:
                json_str = ret.stdout.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
                try:
                    res_data = json.loads(json_str)
                    final_path = res_data.get("output_path")
                    debug_path = res_data.get("debug_path")
                    
                    if final_path and os.path.exists(final_path):
                        st.success(f"Final Summary Saved: `{final_path}`")
                        st.session_state['final_json_path'] = final_path
                        
                        # Display Final Data
                        with open(final_path, 'r') as f:
                            final_json = json.load(f)
                        
                        # --- FORMAT FOR DISPLAY ---
                        df_display = pd.DataFrame(final_json)
                        
                        # Format "Reactants" (list of dicts) -> String
                        if "Reactants" in df_display.columns:
                            def fmt_reactants(x):
                                if isinstance(x, list):
                                    # e.g. "Name (Role)" or just "Name"
                                    return ", ".join([f"{i.get('name','')}" for i in x])
                                return str(x)
                            df_display["Reactants"] = df_display["Reactants"].apply(fmt_reactants)
                        
                        # Format "Products" (list of dicts) -> String
                        if "Products" in df_display.columns:
                            def fmt_products(x):
                                if isinstance(x, list):
                                    # e.g. "Name (Yield)"
                                    items = []
                                    for i in x:
                                        name = i.get('name', '')
                                        yld = i.get('yield', '')
                                        if yld and yld != "0.00%":
                                            items.append(f"{name} ({yld})")
                                        else:
                                            items.append(name)
                                    return ", ".join(items)
                                return str(x)
                            df_display["Products"] = df_display["Products"].apply(fmt_products)

                        st.dataframe(df_display)
                        
                        # Display Debug Info (Step 5a/5b Results)
                        if debug_path and os.path.exists(debug_path):
                            with open(debug_path, 'r') as f:
                                debug_json = json.load(f)
                            
                            st.divider()
                            st.subheader("Step 5 Process Details (Context Resolution)")
                            for item in debug_json:
                                with st.expander(f"Source: {item['source']}", expanded=False):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.markdown("**Step 5a: Global Candidates**")
                                        st.json(item.get("step5a_candidates", {}))
                                    with c2:
                                        st.markdown("**Step 5b: Resolved Context**")
                                        st.json(item.get("step5b_resolved", {}))
                                        if item.get("step5b_unknowns"):
                                            st.caption(f"Unknowns queried: {item.get('step5b_unknowns')}")

                    else:
                        st.error("Output file not found.")
                except Exception as e:
                    st.error(f"Failed to parse result JSON: {e}")
            else:
                 st.warning("Script finished but returned no structured JSON result.")
            
            status.update(label="Global Assembly Complete", state="complete")
    
    # Show Result
    # Check if file exists or in session
    # OUTPUT NAME UPDATE: step5_global_single.py outputs {basename}_final_summary.json
    final_file = os.path.join("data/final_output", f"{pdf_basename}_final_summary.json")
    
    if os.path.exists(final_file):
        st.subheader("Final Dataset")
        st.markdown(f"Path: `{final_file}`")
        
        try:
            with open(final_file, 'r') as f:
                content = f.read()
                # Try parse json
                try:
                    j_data = json.loads(content)
                    st.json(j_data)
                except:
                    st.error("Invalid JSON Content (Likely LLM Error):")
                    st.text(content)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            # It might be raw text or JSON
            content = f.read()
            try:
                j = json.loads(content)
                st.json(j)
            except:
                st.text(content)
