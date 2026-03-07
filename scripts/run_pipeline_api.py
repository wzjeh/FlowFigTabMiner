import os
import sys
import argparse
import subprocess
import time

def run_subprocess(command, phase_name):
    """Run a CLI subprocess and log its execution."""
    print(f"\n{'='*50}")
    print(f"[{phase_name}] Starting execution...")
    print(f"Command: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"[{phase_name}] ERROR: Process failed with return code {result.returncode}")
        sys.exit(result.returncode)
        
    print(f"[{phase_name}] Completed successfully in {time.time() - start_time:.2f}s")
    print(f"{'='*50}\n")

def run_pipeline(pdf_path, output_dir="data/final_output"):
    print(f"--- FlowDevMiner Multi-Process Pipeline API for {pdf_path} ---")
    total_start = time.time()
    
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)
    os.makedirs(output_dir, exist_ok=True)
    
    python_exec = sys.executable
    
    # --- PHASE 1: Document Parsing ---
    cmd1 = f"{python_exec} scripts/phase1_input.py {pdf_path} {intermediate_dir}"
    run_subprocess(cmd1, "Phase 1: Input & Parse")
    
    # --- PHASE 2: Figure Processing ---
    cmd2 = f"{python_exec} scripts/phase2_figures.py {intermediate_dir}"
    run_subprocess(cmd2, "Phase 2: Figures")
    
    # --- PHASE 3: Table Processing ---
    cmd3 = f"{python_exec} scripts/phase3_tables.py {intermediate_dir}"
    run_subprocess(cmd3, "Phase 3: Tables")
    
    # --- PHASE 4: Semantic Synthesis & Export ---
    cmd4 = f"{python_exec} scripts/phase4_synthesis.py {pdf_path} {intermediate_dir} {output_dir}"
    run_subprocess(cmd4, "Phase 4: Synthesis")
    
    print(f"--- Full Pipeline Completed in {time.time() - total_start:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full FlowDevMiner Pipeline (Multi-Process)")
    parser.add_argument("pdf_path", help="Path to input PDF")
    args = parser.parse_args()
    run_pipeline(args.pdf_path)
