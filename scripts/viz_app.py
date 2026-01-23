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

# Sidebar
st.sidebar.header("Configuration")
app_mode = st.sidebar.radio("Mode", ["Step-by-Step (Figures)", "Table Extraction", "Batch Process (One-Click)", "Gallery View", "Upload Single Image"])

# --- MODE: TABLE EXTRACTION ---
if app_mode == "Table Extraction":
    st.header("📊 Table Extraction Debugger")
    
    # Input Selection
    input_type = st.radio("Input Type", ["Select from PDF", "Upload Image"])
    
    selected_image_path = None
    
    if input_type == "Select from PDF":
        input_dir = "data/input"
        # Ensure input dir exists
        if os.path.exists(input_dir):
            pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
            pdf_map = {os.path.basename(f): f for f in pdf_files}
            selected_pdf_name = st.selectbox("Select PDF", list(pdf_map.keys()))
            
            if selected_pdf_name:
                pdf_basename = os.path.splitext(selected_pdf_name)[0]
                tables_dir = os.path.join("data/intermediate", pdf_basename, "tables") # Assuming TF-ID saves here?
                # Check root intermediate dir too if structure differs
                if not os.path.exists(tables_dir):
                     tables_dir = os.path.join("data/intermediate", pdf_basename) # Sometimes just flat?
                
                if os.path.exists(tables_dir):
                     # Recursive search for images with 'table' in name? 
                     # Or just list dir
                     possible_files = glob.glob(os.path.join(tables_dir, "*table*.png")) + glob.glob(os.path.join(tables_dir, "*table*.jpg"))
                     if possible_files:
                         selected_img_name = st.selectbox("Select Table Image", [os.path.basename(f) for f in possible_files])
                         selected_image_path = os.path.join(tables_dir, selected_img_name)
                     else:
                         st.warning(f"No table images found in {tables_dir}. Run Step 1 first.")
                else:
                     st.warning(f"Directory not found: {tables_dir}. Run Step 1 first.")
        else:
             st.error("data/input directory missing.")

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
        st.image(selected_image_path, caption="Original Table", width=600)
        
        if st.button("Run Table Pipeline"):
            with st.spinner("Running Pipeline (Filter -> Structure -> Content)..."):
                # Create output dir for debug crops
                out_debug = os.path.join(os.path.dirname(selected_image_path), "debug_output")
                res = run_script("scripts/step_table_pipeline.py", [selected_image_path, "--output_dir", out_debug])
                
                json_out = parse_json_output(res.stdout)
                
                if json_out:
                    if json_out.get('is_valid'):
                        st.success("Analysis Complete & Valid Table!")
                    else:
                        st.error(f"Invalid / Filtered: {json_out.get('reason')}")
                    
                    st.json(json_out)
                    
                    # Show DataFrame
                    if 'dataframe' in json_out:
                         st.subheader("Extracted Data")
                         st.dataframe(pd.DataFrame(json_out['dataframe']))
                    
                    # Show Cells Grid if available (Optional visualization)
                    st.subheader("Detected Cells")
                    cells = json_out.get('cells', [])
                    if cells:
                        st.write(f"Count: {len(cells)}")
                        # Maybe show first 5 crops?
                        cols = st.columns(5)
                        for i, cell in enumerate(cells[:10]):
                            with cols[i%5]:
                                if 'crop_path' in cell and os.path.exists(cell['crop_path']):
                                    st.image(cell['crop_path'], caption=f"R{cell.get('row_index')}C{cell.get('col_index')}\n{cell.get('class')}")
                else:
                    st.error(f"Pipeline Failed: {res.stdout}")
                    st.text(res.stderr)
    st.header("📤 Upload Single Image")
    st.info("Note: Uploaded images only support Steps 2-4 (No Step 5 LLM).")
    uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        # Save to temp
        upload_dir = "data/intermediate/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Saved to {file_path}")
        st.image(file_path, caption="Uploaded Image", width=500)
        
        # Step 2: Macro Cleaning
        st.subheader("Step 2: Macro Cleaning")
        if st.button("Run Step 2 (Clean)"):
            with st.spinner("Running YOLO Macro..."):
                res = run_script("scripts/step2_macro_single.py", [file_path])
                json_out = parse_json_output(res.stdout)
                if json_out and 'cleaned_image' in json_out:
                    st.session_state['step2_result_upload'] = json_out
                    st.success("Cleaning Complete.")
                else:
                    st.error(f"Step 2 Error: {res.stdout}")
        
        if 'step2_result_upload' in st.session_state:
            res2 = st.session_state['step2_result_upload']
            cleaned_path = res2['cleaned_image']
            
            c1, c2 = st.columns(2)
            c1.image(res2['raw_image'], caption="Raw", width=300)
            c2.image(cleaned_path, caption="Cleaned", width=300)
            
            # Display Detected Elements (Crops) - 5 Column Grid
            st.subheader("Extracted Elements")
            elements = res2.get('elements', {})
            if elements:
                folder_path = os.path.dirname(cleaned_path)
                if st.button(f"📂 Open Folder", key="open_folder_upload"):
                    subprocess.run(["open", folder_path])

                # Separate Caption to display at bottom
                caption_paths = elements.pop("caption", [])
                
                # Display Standard Elements (including 'chart_text' which is now above/middle)
                for label, paths in elements.items():
                    if isinstance(paths, list) and paths:
                        st.markdown(f"**{label.replace('_', ' ').title()}**")
                        if "y_axis" in label:
                             cols = st.columns(5)
                             mod_val = 5
                        else:
                             cols = st.columns(3) # Wider columns for Legend/X-axis
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
                                             st.image(p, caption=os.path.basename(p), width=100) # Fallback
                                    else:
                                         # Others: Width = 300
                                         st.image(p, caption=os.path.basename(p), width=300)
                                else:
                                    st.error(f"Missing: {os.path.basename(p)}")

                # Display Caption at Bottom (Wider)
                if caption_paths:
                    st.markdown("---")
                    st.markdown("**Chart Caption (Bottom)**")
                    cols = st.columns(1) 
                    # width=600 in 1 column is safe.
                    for p in caption_paths:
                        if os.path.exists(p):
                             st.image(p, caption=os.path.basename(p), width=600)
            
            # Step 3
            st.subheader("Step 3: Micro Detection")
            if st.button("Run Step 3 (Detect)"):
                with st.spinner("Running Detection..."):
                    res = run_script("scripts/step3_micro_single.py", [cleaned_path])
                    json_out = parse_json_output(res.stdout)
                    if json_out:
                        st.session_state['step3_result_upload'] = json_out
                        st.success("Detection Complete.")
                    else:
                        st.error(f"Step 3 Error: {res.stdout}")

            if 'step3_result_upload' in st.session_state:
                res3 = st.session_state['step3_result_upload']
                st.metric("Points Detected", res3['num_points_detected'])
                st.dataframe(pd.DataFrame(res3['mapped_data']))
                
                # Step 4
                st.subheader("Step 4: Assembly")
                if st.button("Run Step 4"):
                    temp_json = "temp_extraction_upload.json"
                    with open(temp_json, 'w') as f:
                        json.dump(res3['mapped_data'], f)
                        
                    fid = os.path.splitext(os.path.basename(cleaned_path))[0].replace("_cleaned", "")
                    idir = os.path.dirname(cleaned_path)
                    
                    # Pass extracted caption to assembly
                    res = run_script("scripts/step4_assembly_single.py", [fid, idir, temp_json])
                    json_out = parse_json_output(res.stdout)
                    if json_out:
                        st.success("Assembly Complete")
                        st.json(json_out['content'])
                    else:
                        st.warning("Filtered out.")

# --- PDF MODES ---
elif app_mode != "Table Extraction": # Handle other modes
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
        if app_mode == "Gallery View":
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
        elif app_mode == "Batch Process (One-Click)":
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
        elif app_mode == "Step-by-Step (Figures)":
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
