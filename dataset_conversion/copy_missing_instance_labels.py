import os
import shutil
from tqdm import tqdm

def main():
    source_dir = "/mnt/home2/liuyujie/data/NPC_MRI/raw_data/labels"
    target_dir = "/mnt/home2/liuyujie/data/NPC_MRI/NPC_data_V20260102"
    target_filename = "flip_T2_ax_lymph_instance.nii.gz"

    if not os.path.exists(target_dir):
        print(f"Error: Target directory not found: {target_dir}")
        return
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    cases = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    
    missing_in_source = 0
    copied_count = 0
    deleted_count = 0

    print(f"Found {len(cases)} case folders in {target_dir}")

    for case_id in tqdm(cases):
        case_target_dir = os.path.join(target_dir, case_id)
        target_file_path = os.path.join(case_target_dir, target_filename)
        
        # 1. Delete existing file if present
        if os.path.exists(target_file_path):
            os.remove(target_file_path)
            deleted_count += 1

        # 2. Check source file
        source_file_path = os.path.join(source_dir, f"{case_id}.nii.gz")
        
        if os.path.exists(source_file_path):
            # 3. Copy new file
            shutil.copy2(source_file_path, target_file_path)
            copied_count += 1
        else:
            missing_in_source += 1

    print("\n--- Operation Summary ---")
    print(f"Total Cases Checked: {len(cases)}")
    print(f"Old Files Deleted: {deleted_count}")
    print(f"New Files Copied: {copied_count}")
    if missing_in_source > 0:
        print(f"Warning: {missing_in_source} cases did not have a corresponding file in the source directory.")

if __name__ == "__main__":
    main()
