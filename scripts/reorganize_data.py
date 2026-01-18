import os
import shutil
import glob

def reorganize_directory(base_dir):
    """
    Reorganizes the directory structure:
    FROM: base_dir/{subfolder}/{type}/{image}.png
    TO:   base_dir/{type}/{image}.png
    
    where {type} is 'tables' or 'figures'.
    """
    print(f"Reorganizing {base_dir}...")
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Ensure target directories exist
    target_tables_dir = os.path.join(base_dir, "tables")
    target_figures_dir = os.path.join(base_dir, "figures")
    
    os.makedirs(target_tables_dir, exist_ok=True)
    os.makedirs(target_figures_dir, exist_ok=True)

    # List all subdirectories (excluding the target dirs themselves if they already exist)
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) 
               and d not in ["tables", "figures", ".DS_Store", "yolo_cleaned", "lab pub"]]

    count_moved = 0
    dirs_removed = 0

    for subdir_name in subdirs:
        subdir_path = os.path.join(base_dir, subdir_name)
        
        # Process tables
        src_tables = os.path.join(subdir_path, "tables")
        if os.path.exists(src_tables):
            for f in os.listdir(src_tables):
                src_file = os.path.join(src_tables, f)
                # To avoid name collisions, we might want to prefix with the subdir name if needed
                # But typically filenames like page_X_figure_Y.png are generic. 
                # Ideally they should be unique PER PDF. 
                # If we flatten, we MUST prefix with the PDF name (subdir_name) to avoid collisions.
                # The previous script created filenames like "page_X_figure_Y.png".
                # So we rename to "{subdir_name}_page_X_table_Y.png"
                
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    new_filename = f"{subdir_name}_{f}"
                    dst_file = os.path.join(target_tables_dir, new_filename)
                    shutil.move(src_file, dst_file)
                    count_moved += 1
        
        # Process figures
        src_figures = os.path.join(subdir_path, "figures")
        if os.path.exists(src_figures):
            for f in os.listdir(src_figures):
                 src_file = os.path.join(src_figures, f)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    new_filename = f"{subdir_name}_{f}"
                    dst_file = os.path.join(target_figures_dir, new_filename)
                    shutil.move(src_file, dst_file)
                    count_moved += 1
        
        # Remove the now empty (or processed) subdirectory
        # We can use shutil.rmtree but let's be safe and check if we moved everything we cared about
        # For this task, the user said "remove subfolders created", so we remove the whole subdir
        shutil.rmtree(subdir_path)
        dirs_removed += 1

    print(f"Moved {count_moved} files.")
    print(f"Removed {dirs_removed} subdirectories.")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intermediate_dir = os.path.join(project_root, "data", "intermediate")
    lab_pub_dir = os.path.join(intermediate_dir, "lab pub")

    # 1. Reorganize 'lab pub'
    reorganize_directory(lab_pub_dir)

    # 2. Reorganize 'intermediate' (only numbered folders essentially, excluded lab pub inside the function)
    # The function exclude list handles 'lab pub' so it won't be processed as a subdir of intermediate
    reorganize_directory(intermediate_dir)
    
if __name__ == "__main__":
    main()
