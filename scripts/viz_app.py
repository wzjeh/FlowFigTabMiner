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
            
            # Create a dedicated output directory for this table's components
            # Structure: data/intermediate/{pdf_name}/tables/{table_basename}/
            if input_type == "Select from PDF Extraction":
                # data/intermediate/{pdf_name}/tables/{table_img_name}/
                base_dir = os.path.dirname(selected_image_path)
                table_name = os.path.splitext(os.path.basename(selected_image_path))[0]
                out_dir = os.path.join(base_dir, table_name)
            else:
                out_dir = os.path.join(os.path.dirname(selected_image_path), "processed", os.path.splitext(uploaded_file.name)[0])
            
            # --- STEP 2: SEGMENTATION ---
            st.subheader("Step 2: Table Segmentation (YOLOv11)")
            
            if st.button("Run Step 2 (Segmentation)"):
                with st.spinner("Running YOLOv11..."):
                    res = run_script("scripts/step_table_segmentation.py", [selected_image_path, "--output_dir", out_dir])
                    json_out = parse_json_output(res.stdout)
                    
                    if json_out:
                        st.session_state['table_step2'] = json_out
                        if json_out.get('is_table'):
                            st.success("Table Detected & Segmented")
                        else:
                            st.warning("No Table Body detected.")
                    else:
                        st.error("Step 2 Failed.")
                        with st.expander("Log"):
                            st.text(res.stdout)
                            st.text(res.stderr)

            if 'table_step2' in st.session_state:
                res2 = st.session_state['table_step2']
                
                # Show Logs
                with st.expander("Step 2 Debug Logs (Raw Detections)"):
                    if 'logs' in res2:
                        st.dataframe(pd.DataFrame(res2['logs']))
                    else:
                        st.info("No detailed logs.")

                # Show Images
                # Components found
                c1, c2, c3, c4 = st.columns(4)
                
                # Helper to find saved images from logs or disk
                def get_saved_paths(label):
                    return [l['saved_path'] for l in res2.get('logs', []) if l['label'] == label]

                with c1:
                    st.markdown("**Caption**")
                    for p in get_saved_paths("table_caption"): st.image(p)
                with c2:
                    st.markdown("**Table Body (Main)**")
                    # Main Body
                    main_body = res2.get("best_body_crop_path")
                    if main_body and os.path.exists(main_body):
                        st.image(main_body, caption="Processed Crop (Full Width + Padding)")
                    else:
                        for p in get_saved_paths("table_body"): st.image(p, caption="Raw Crop")
                with c3:
                    st.markdown("**Note**")
                    for p in get_saved_paths("table_note"): st.image(p)
                with c4:
                    st.markdown("**Scheme**")
                    for p in get_saved_paths("table_scheme"): st.image(p)

                # --- STEP 3: STRUCTURE RECOGNITION ---
                st.subheader("Step 3: Structure Recognition (TATR)")
                
                # Only enable if body exists
                body_path = res2.get("best_body_crop_path")
                
                if body_path and os.path.exists(body_path):
                    if st.button("Run Step 3 (Structure)"):
                        with st.spinner("Running Table Transformer..."):
                            res = run_script("scripts/step_table_structure.py", [body_path])
                            json_out = parse_json_output(res.stdout)
                            if json_out:
                                st.session_state['table_step3'] = json_out
                                st.success("Structure Recognized")
                            else:
                                st.error("Step 3 Failed.")
                                with st.expander("Log"):
                                    st.text(res.stdout)
                                    st.text(res.stderr)
                    
                    if 'table_step3' in st.session_state:
                        res3 = st.session_state['table_step3']
                        
                        # Show Viz
                        viz_path = res3.get("viz_path")
                        if viz_path and os.path.exists(viz_path):
                            # Fix warning: use_container_width -> use_column_width for compatibility or width='stretch'
                            # Sticking to use_column_width=True which is generally safe or ignore warning for now.
                            # But user specifically asked to fix it.
                            # Streamlit warning says: use width='stretch'
                            st.image(viz_path, caption="TATR Detection (Green=Row, Orange=Col, Red=Cell)", width="stretch") 
                        
                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Rows", res3.get('num_rows', 0))
                        m2.metric("Columns", res3.get('num_columns', 0))
                        m3.metric("Cells", res3.get('num_cells', 0))
                        
                        # Logs
                        with st.expander("Step 3 Debug Logs (Detected Objects)"):
                            if 'logs' in res3:
                                st.dataframe(pd.DataFrame(res3['logs']))
                                
                        # --- STEP 4: CONTENT EXTRACTION (OCR) ---
                        st.subheader("Step 4: Content Extraction (OCR -> CSV)")
                        
                        # We need the structure logs which contain the cell boxes
                        # Where is the structure JSON saved? 
                        # step_table_structure.py outputted a JSON to stdout, but we need to save it to disk for the next script to read
                        # or pass it. step_table_assembly.py takes "structure_json_path".
                        # We need to save the JSON from Step 3 to a file if it wasn't already.
                        # Actually, let's just save st.session_state['table_step3'] to a temp file or proper path.
                        
                        structure_json_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(body_path))[0]}_structure.json")
                        
                        if st.button("Run Step 4 (OCR Assembly)"):
                            # Save Structure JSON first
                            with open(structure_json_path, 'w') as f:
                                json.dump(res3, f) # res3 is the Step 3 output dict
                            
                            with st.spinner("Running OCR & Assembly..."):
                                # Call Step 4 script
                                # Usage: step_table_assembly.py body_image_path structure_json_path --output_dir ...
                                res4 = run_script("scripts/step_table_assembly.py", [
                                    body_path, 
                                    structure_json_path, 
                                    "--output_dir", out_dir,
                                    "--padding", "14" # optimized white padding
                                ])
                                
                                # Store raw output for log display
                                st.session_state['table_step4_raw_stdout'] = res4.stdout
                                st.session_state['table_step4_raw_stderr'] = res4.stderr
                                
                                json_out4 = parse_json_output(res4.stdout)
                                
                                if json_out4:
                                    st.session_state['table_step4'] = json_out4
                                    st.success("CSV Generated!")
                                else:
                                    st.error("Step 4 Failed.")
                        
                        # Display Functionality regardless of success to show logs
                        if 'table_step4_raw_stdout' in st.session_state:
                            st.markdown("### Execution Logs")
                            
                            # Filter for interesting lines (Cell X...)
                            logs = st.session_state['table_step4_raw_stdout']
                            cell_logs = [line for line in logs.split('\n') if "Cell " in line and "]:" in line]
                            
                            if cell_logs:
                                with st.expander("OCR Detail Logs", expanded=True):
                                    st.code("\n".join(cell_logs))
                            else:
                                with st.expander("Full stdout"):
                                    st.text(logs)
                            
                            if st.session_state.get('table_step4_raw_stderr'):
                                with st.expander("Stderr"):
                                    st.text(st.session_state['table_step4_raw_stderr'])

                        if 'table_step4' in st.session_state:
                            res4 = st.session_state['table_step4']
                            csv_path = res4.get('csv_path')
                            
                            if csv_path and os.path.exists(csv_path):
                                st.markdown(f"**Saved CSV:** `{os.path.basename(csv_path)}`")
                                df = pd.read_csv(csv_path, header=None)
                                st.dataframe(df)
                                
                            # Debug: Extracted Cells Gallery
                            cells_debug_dir = os.path.join(out_dir, "cells_debug")
                            if os.path.exists(cells_debug_dir):
                                st.markdown("### Debug: Cell Crops")
                                cell_imgs = sorted(glob.glob(os.path.join(cells_debug_dir, "*.png")))[:20] # Show first 20
                                if cell_imgs:
                                    cols = st.columns(5)
                                    for i, p in enumerate(cell_imgs):
                                        with cols[i % 5]:
                                            st.image(p, caption=os.path.basename(p), width=100)
                                else:
                                    st.info("No cell crops found.")
                            else:
                                st.info(f"No cells_debug dir at {cells_debug_dir}")
                        
                else:
                    st.warning("Cannot proceed to Step 3: No valid table body found.")

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
                        
                        # Heuristic Auto-Detection for Heatmap
                        default_heatmap = False
                        if os.path.exists(cleaned_path):
                            try:
                                import cv2
                                import numpy as np
                                
                                # Check background fill ratio
                                # Standard plots are mostly white. Heatmaps are mostly colored/gray.
                                img_check = cv2.imread(cleaned_path, cv2.IMREAD_GRAYSCALE)
                                if img_check is not None:
                                    # Count white pixels (> 240)
                                    white_pixels = np.sum(img_check > 240)
                                    total_pixels = img_check.size
                                    white_ratio = white_pixels / total_pixels
                                    
                                    # If less than 60% white, likely a heatmap or filled contour
                                    if white_ratio < 0.60:
                                        default_heatmap = True
                            except: pass

                        col_opt1, col_opt2 = st.columns(2)
                        use_log_x = col_opt1.checkbox("Log Scale X-Axis", value=False)
                        extract_labels = col_opt2.checkbox("Extract Point Labels (Heatmap)", value=default_heatmap)
                        
                        if st.button("Run Step 3"):
                             with st.spinner("Detecting..."):
                                args = [cleaned_path]
                                if use_log_x: args.append("--log_x")
                                if extract_labels: args.append("--extract_labels")
                                
                                if extract_labels: args.append("--extract_labels")
                                
                                proc = run_script("scripts/step3_micro_single.py", args)
                                json_out = parse_json_output(proc.stdout)
                                if json_out:
                                    st.session_state['step3_result'] = json_out
                                    st.success("Done.")
                        
                        if 'step3_result' in st.session_state:
                            res = st.session_state['step3_result']
                            st.metric("Points", res.get("num_points_detected", 0))
                            
                            classes = res.get("detected_classes", [])
                            if classes:
                                st.write(f"**Detected Classes:** `{', '.join(classes)}`")
                            
                            # Visualization
                            dets = res.get("detections", [])
                            if dets and os.path.exists(cleaned_path):
                                import cv2
                                import numpy as np
                                vis_img = cv2.imread(cleaned_path)
                                if vis_img is not None:
                                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                    for d in dets:
                                        bbox = list(map(int, d['box']))
                                        label = d['label']
                                        color = (255, 0, 0) # Red for points
                                        if 'point' not in label: color = (0, 255, 0) # Green for text
                                        if 'value' in label: color = (0, 165, 255) # Orange for values
                                        
                                        if 'point' in label or 'marker' in label:
                                            cx, cy = d['center']
                                            cv2.circle(vis_img, (int(cx), int(cy)), 4, color, -1)
                                        else:
                                            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                                            cv2.putText(vis_img, label, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    
                                    
                                    # Fix: 'use_container_width' is deprecated in strict versions, use width='stretch'
                                    try:
                                        st.image(vis_img, caption="YOLO Detections (Red=Points, Green=Text, Orange=Values)", width="stretch")
                                    except:
                                        # Fallback for older Streamlit
                                        st.image(vis_img, caption="YOLO Detections (Red=Points, Green=Text, Orange=Values)", use_column_width=True)
                            
                            debug_logs = res.get("debug_log", [])
                            if debug_logs:
                                with st.expander("Debug Logs (Coordinate Mapper)"):
                                    for l in debug_logs:
                                        st.text(l)

    
                            
                            with st.expander("Debug: Full Result JSON"):
                                st.json(res)
                                
                            data = res.get("mapped_data", [])
                            if data:
                                df = pd.DataFrame(data)
                                st.dataframe(df)
                            else:
                                st.warning("No data mapped. (Check if Axis Labels or Data Values were detected)")
                             
                            # Step 4
                            st.subheader("Step 4: Assembly")
                            if st.button("Run Step 4"):
                                temp_json = "temp_extraction.json"
                                with open(temp_json, 'w') as f:
                                    json.dump(res['mapped_data'], f)
                                fid = os.path.splitext(os.path.basename(cleaned_path))[0].replace("_cleaned", "")
                                idir = os.path.dirname(cleaned_path)
                                res_s4 = run_script("scripts/step4_assembly_single.py", [fid, idir, temp_json])
                                json_out_s4 = parse_json_output(res_s4.stdout)
                                
                                # Show logs for debugging "Filtered out"
                                with st.expander("Step 4 Execution Log (Debug)"):
                                    st.text(res_s4.stdout)
                                    st.text(res_s4.stderr)
                                
                                if json_out_s4:
                                    st.success("Saved.")
                                    st.json(json_out_s4['content'])
                                else:
                                    st.warning("Filtered out.")
                 
                                    
    else:
        st.info("Select a PDF.")
