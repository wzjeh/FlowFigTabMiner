import streamlit as st
import os
import glob
import subprocess
import json
import pandas as pd
import time
from PIL import Image

st.set_page_config(layout="wide", page_title="FlowFigTabMiner Pipeline Viz")

# --- Utils ---
PYTHON_EXEC = "./flowfigtabminer/bin/python3"

def run_script(script_path, args=[]):
    cmd = [PYTHON_EXEC, script_path] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def parse_json_output(stdout):
    try:
        if "---JSON_START---" in stdout:
            json_str = stdout.split("---JSON_START---")[1].split("---JSON_END---")[0]
            return json.loads(json_str)
    except:
        pass
    return None

# --- UI ---
st.title("🧪 FlowFigTabMiner Pipeline Visualization")

# Sidebar Header
st.sidebar.header("Pipeline Module")
module = st.sidebar.radio("Select Module", ["Figure Extraction", "Table Extraction"])

# --- TABLE EXTRACTION MODULE ---
if module == "Table Extraction":
    st.sidebar.markdown("---")
    st.sidebar.header("Table Options")
    table_mode = st.sidebar.radio("Activity", ["Step-by-Step (Single)", "Batch Processing (TODO)", "Gallery View (TODO)"])
    
    # === TABLE: STEP-BY-STEP (SINGLE) ===
    if table_mode == "Step-by-Step (Single)":
        st.header("📊 Table Extraction: Step-by-Step")
        
        # Input Selection
        input_type = st.radio("Input Source", ["Select from PDF Extraction", "Upload Image"])
        
        selected_image_path = None
        
        if input_type == "Select from PDF Extraction":
            input_dir = "data/input"
            if os.path.exists(input_dir):
                pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
                pdf_map = {os.path.basename(f): f for f in pdf_files}
                selected_pdf_name = st.selectbox("Select PDF", list(pdf_map.keys()))
                
                if selected_pdf_name:
                    pdf_basename = os.path.splitext(selected_pdf_name)[0]
                    # Try finding tables in intermediate dir
                    # Structure: data/intermediate/{pdf_name}/tables/
                    tables_dir = os.path.join("data/intermediate", pdf_basename, "tables")
                    # Fallback check
                    if not os.path.exists(tables_dir):
                         tables_dir = os.path.join("data/intermediate", pdf_basename)
                    
                    if os.path.exists(tables_dir):
                         possible_files = glob.glob(os.path.join(tables_dir, "*table*.png")) + glob.glob(os.path.join(tables_dir, "*table*.jpg"))
                         # Exclude debug crops if any (avoid recursion)
                         possible_files = [f for f in possible_files if "_body" not in f and "_caption" not in f]
                         
                         if possible_files:
                             selected_img_name = st.selectbox("Select Table", [os.path.basename(f) for f in possible_files])
                             selected_image_path = os.path.join(tables_dir, selected_img_name)
                         else:
                             st.warning("No table images found. Run Step 1 (TF-ID) first.")
                    else:
                         st.warning("Intermediate folder not found.")
            else:
                 st.error("Data input directory missing.")

        else:
            uploaded_file = st.file_uploader("Upload Table Image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                upload_dir = "data/intermediate/uploads_tables"
                os.makedirs(upload_dir, exist_ok=True)
                selected_image_path = os.path.join(upload_dir, uploaded_file.name)
                with open(selected_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        # Main Processing UI
        if selected_image_path:
            st.image(selected_image_path, caption="Original Table Image", width=600)
            
            # Create a dedicated output directory for this table's components
            # Structure: data/intermediate/{pdf_name}/tables/{table_basename}/
            # For uploads, use upload_dir/processed/{name}
            if input_type == "Select from PDF Extraction":
                # data/intermediate/{pdf_name}/tables/{table_img_name}/
                base_dir = os.path.dirname(selected_image_path)
                table_name = os.path.splitext(os.path.basename(selected_image_path))[0]
                # Ensure we don't nest infinitely if already in a subfolder
                # If selected_image_path is .../tables/img.png -> .../tables/img/
                out_dir = os.path.join(base_dir, table_name)
            else:
                out_dir = os.path.join(os.path.dirname(selected_image_path), "processed", os.path.splitext(uploaded_file.name)[0])
            
            if st.button("Run Extraction Pipeline"):
                with st.spinner("Running Pipeline..."):
                    # Pass the *parent* directory or specific output dir?
                    # The script step_table_pipeline.py takes --output_dir.
                    # TablePipeline.process_table uses output_dir to create subfolder based on image name IF it's a base dir.
                    # BUT we updated TablePipeline to append basename if output_dir is provided.
                    # so if we pass .../tables/, it creates .../tables/table1/
                    # Here we want to control it. Let's pass the parent of the target folder.
                    # Actually, let's pass the exact parent dir where we want the folder to be created.
                    target_parent_dir = os.path.dirname(out_dir)
                    
                    res = run_script("scripts/step_table_pipeline.py", [selected_image_path, "--output_dir", target_parent_dir])
                    json_out = parse_json_output(res.stdout)
                    
                    if json_out:
                        st.session_state['table_result'] = json_out
                        if json_out.get('is_valid'):
                            st.success("Extraction Successful!")
                        else:
                            st.error(f"Extraction Failed/Filtered: {json_out.get('reason')}")
                    else:
                        st.error("Pipeline execution failed.")
                        with st.expander("Log"):
                            st.text(res.stdout)
                            st.text(res.stderr)

            if 'table_result' in st.session_state:
                res = st.session_state['table_result']
                
                # Check for extracted components
                # We expect them in out_dir
                if os.path.exists(out_dir):
                    st.subheader("1. Segmentation Results (YOLOv11)")
                    
                    # Layout: 4 columns for components
                    c1, c2, c3, c4 = st.columns(4)
                    
                    # Helper to find images
                    def get_comp_img(label):
                        # Pattern: {table_basename}_{label}_*.png
                        # json_out might have paths? No, strictly saved to disk.
                        # Let's search pattern
                        table_basename = os.path.splitext(os.path.basename(selected_image_path))[0]
                        pattern = os.path.join(out_dir, f"{table_basename}_{label}_*.png")
                        files = glob.glob(pattern)
                        return files
                    
                    with c1:
                        st.markdown("**Caption**")
                        imgs = get_comp_img("table_caption")
                        for img in imgs: st.image(img)
                        
                    with c2:
                        st.markdown("**Subject/Body**")
                        # Look for main body
                        # pattern: table_basename_body_main.png OR table_caption_table_body_*.png
                        # In pipeline we saved: {table_basename}_body_main.png
                        main_body = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(selected_image_path))[0]}_body_main.png")
                        if os.path.exists(main_body):
                            st.image(main_body)
                        else:
                            # Fallback to parts
                            imgs = get_comp_img("table_body")
                            for img in imgs: st.image(img)

                    with c3:
                        st.markdown("**Note**")
                        imgs = get_comp_img("table_note")
                        for img in imgs: st.image(img)
                        
                    with c4:
                        st.markdown("**Scheme**")
                        imgs = get_comp_img("table_scheme") # or table_scheme
                        for img in imgs: st.image(img)

                # Transformer Result (Grid)
                if 'cells' in res:
                    st.subheader("2. Structure Recognition (Transformer)")
                    st.json(res['structure']) # Show raw structure if needed
                    st.write(f"Detected Cells: {len(res['cells'])}")
                
                # CSV Result
                if 'csv_path' in res and res['csv_path'] and os.path.exists(res['csv_path']):
                    st.subheader("3. Extracted Data")
                    df = pd.read_csv(res['csv_path'], header=None)
                    st.dataframe(df)

# --- FIGURE EXTRACTION MODULE ---
elif module == "Figure Extraction":
    st.sidebar.markdown("---")
    st.sidebar.header("Figure Options")
    fig_mode = st.sidebar.radio("Activity", ["Step-by-Step", "Batch Process (One-Click)", "Gallery View", "Upload Single Image"])

    # --- MODE: UPLOAD SINGLE IMAGE ---
    if fig_mode == "Upload Single Image":
        pass # Placeholder for now or specific upload logic
        # Actually it seems the previous code structure intended to fall through?
        # Let's fix the indentation of the following block if it belongs to 'else'
        # Looking at original code, lines 203+ seem to handle the PDF selection logic generally
        # But here they are seemingly unindented relative to 'if' but inside 'elif module == Figure'
    
    # Common PDF Selection Logic (if not upload mode)
    if fig_mode != "Upload Single Image":
        input_dir = "data/input"
        pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
        pdf_map = {os.path.basename(f): f for f in pdf_files}
        
        selected_pdf_name = st.sidebar.selectbox("Select PDF", list(pdf_map.keys()))
        
        if selected_pdf_name:
            selected_pdf = pdf_map[selected_pdf_name]
            basename = os.path.splitext(selected_pdf_name)[0]
            intermediate_dir = os.path.join("data/intermediate", basename)
            evidence_dir = "data/evidence"
        
        # --- GALLERY VIEW ---
        if fig_mode == "Gallery View":
            st.header(f"🖼️ Gallery: {selected_pdf_name}")
            figures_path = os.path.join(intermediate_dir, "figures")
            
            if os.path.exists(figures_path):
                figure_crops = sorted(glob.glob(os.path.join(figures_path, "*.png")))
                if figure_crops:
                    cols = st.columns(3)
                    for i, crop in enumerate(figure_crops):
                        with cols[i % 3]:
                            st.image(crop, caption=os.path.basename(crop), use_container_width=True)
                else:
                    st.info("No figures found. Run Step 1 first.")
            else:
                 st.info(f"No intermediate data found at {figures_path}. Run Step 1 first.")

        # --- BATCH MODE ---
        elif fig_mode == "Batch Process (One-Click)":
            st.header(f"Batch Processing: {selected_pdf_name}")
            if st.button("🚀 Run Full Pipeline"):
                with st.spinner(f"Processing {selected_pdf_name}..."):
                    st.info("Running Step 1 (TF-ID)...")
                    res1 = run_script("scripts/step1_tfid.py", [selected_pdf])
                    if res1.returncode != 0:
                        st.error(f"Step 1 Failed:\n{res1.stderr}")
                    else:
                        st.success("Step 1 Complete.")
                        st.info("Running Steps 2-4...")
                        res2 = run_script("scripts/run_steps2_to_4.py", [selected_pdf])
                        if res2.returncode != 0:
                            st.error(f"Steps 2-4 Failed:\n{res2.stderr}")
                        else:
                            st.success("Pipeline Complete!")
                            with st.expander("Show Log"):
                                st.text(res2.stdout)

            # Show Result Evidence
            st.subheader("Generated Evidence")
            evidence_files = sorted(glob.glob(os.path.join(evidence_dir, "*.json")))
            if evidence_files:
                ev_file = st.selectbox("Select Evidence", [os.path.basename(f) for f in evidence_files])
                if ev_file:
                    full_path = os.path.join(evidence_dir, ev_file)
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    c1, c2 = st.columns(2)
                    c1.json(data['meta'])
                    c2.dataframe(pd.DataFrame(data['raw_data']))
            else:
                st.info("No evidence files found.")

        # --- STEP-BY-STEP MODE ---
        elif fig_mode == "Step-by-Step":
            st.header(f"Step-by-Step: {selected_pdf_name}")
            
            # Step 1
            st.subheader("Step 1: TF-ID Detection")
            if st.button("Run Step 1"):
                with st.spinner("Running TF-ID..."):
                    res = run_script("scripts/step1_tfid.py", [selected_pdf])
                    if res.returncode == 0:
                        st.success("Complete.")
                    else:
                        st.error(res.stderr)
            
            figures_path = os.path.join(intermediate_dir, "figures")
            if os.path.exists(figures_path):
                figure_crops = sorted(glob.glob(os.path.join(figures_path, "*.png")))
                if figure_crops:
                    selected_crop_name = st.selectbox("Select Figure", [os.path.basename(f) for f in figure_crops])
                    selected_crop = os.path.join(figures_path, selected_crop_name)
                    st.image(selected_crop, caption="Original Crop", width=500)
                    
                    # Step 2
                    st.subheader("Step 2: Macro Cleaning")
                    if st.button("Run Step 2 (Clean)"):
                        with st.spinner("Cleaning..."):
                            res = run_script("scripts/step2_macro_single.py", [selected_crop])
                            json_out = parse_json_output(res.stdout)
                            if json_out and 'cleaned_image' in json_out:
                                st.session_state['step2_result'] = json_out
                                st.success("Done.")
                            else:
                                st.error(res.stdout)
                    
                    if 'step2_result' in st.session_state:
                        res2 = st.session_state['step2_result']
                        cleaned_path = res2['cleaned_image']
                        
                        c1, c2 = st.columns(2)
                        c1.image(res2['raw_image'], caption="Raw", width=300)
                        c2.image(cleaned_path, caption="Cleaned", width=300)
                        
                        # Elements Gallery (Updated)
                        st.subheader("Extracted Elements")
                        elements = res2.get('elements', {})
                        if elements:
                            folder_path = os.path.dirname(cleaned_path)
                            if st.button(f"📂 Open Folder", key="open_folder_step"):
                                subprocess.run(["open", folder_path])
                            
                            # Separate Caption
                            caption_paths = elements.pop("caption", [])

                            for label, paths in elements.items():
                                if isinstance(paths, list) and paths:
                                    st.markdown(f"**{label.replace('_', ' ').title()}**")
                                    if "y_axis" in label:
                                         cols = st.columns(5)
                                         mod_val = 5
                                    else:
                                         cols = st.columns(3)
                                         mod_val = 3
                                    
                                    for i, p in enumerate(paths):
                                        with cols[i % mod_val]:
                                            if os.path.exists(p):
                                                 # Custom Sizing Logic
                                                if "y_axis" in label:
                                                     # Enforce Height = 500
                                                     try:
                                                         img = Image.open(p)
                                                         w, h = img.size
                                                         target_h = 500
                                                         target_w = int(w * (target_h / h))
                                                         st.image(p, caption=os.path.basename(p), width=target_w)
                                                     except:
                                                         st.image(p, caption=os.path.basename(p), width=100)
                                                else:
                                                     # Others: Width = 300
                                                     st.image(p, caption=os.path.basename(p), width=300)
                            
                            # Display Caption at Bottom
                            if caption_paths:
                                st.markdown("---")
                                st.markdown("**Chart Caption (Bottom)**")
                                for p in caption_paths:
                                    if os.path.exists(p):
                                         st.image(p, caption=os.path.basename(p), width=600)
                        
                        # Step 3
                        st.subheader("Step 3: Detection")
                        if st.button("Run Step 3"):
                             with st.spinner("Detecting..."):
                                res = run_script("scripts/step3_micro_single.py", [cleaned_path])
                                json_out = parse_json_output(res.stdout)
                                if json_out:
                                    st.session_state['step3_result'] = json_out
                                    st.success("Done.")
                        
                        if 'step3_result' in st.session_state:
                             res3 = st.session_state['step3_result']
                             st.metric("Points", res3['num_points_detected'])
                             st.dataframe(pd.DataFrame(res3['mapped_data']))
                             
                             # Step 4
                             st.subheader("Step 4: Assembly")
                             if st.button("Run Step 4"):
                                 temp_json = "temp_extraction.json"
                                 with open(temp_json, 'w') as f:
                                     json.dump(res3['mapped_data'], f)
                                 fid = os.path.splitext(os.path.basename(cleaned_path))[0].replace("_cleaned", "")
                                 idir = os.path.dirname(cleaned_path)
                                 res = run_script("scripts/step4_assembly_single.py", [fid, idir, temp_json])
                                 json_out = parse_json_output(res.stdout)
                                 if json_out:
                                     st.success("Saved.")
                                     st.json(json_out['content'])
                                 else:
                                     st.warning("Filtered out.")
                else:
                    st.info("No crops found.")
    else:
        st.info("Select a PDF.")
