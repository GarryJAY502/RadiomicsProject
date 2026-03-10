import os
import SimpleITK as sitk
from tqdm import tqdm
import glob

def find_and_convert_nrrd_to_nii(dataset_dir: str):
    """
    Scans the dataset directory for .nrrd files.
    If found, it reads the image and saves it as a .nii.gz file using the expected naming convention.
    The original .nrrd file is kept.
    """
    print(f"Scanning {dataset_dir} for .nrrd files to convert...")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} not found.")
        return
        
    cases = sorted(os.listdir(dataset_dir))
    converted_count = 0
    
    for case_id in tqdm(cases):
        case_dir = os.path.join(dataset_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
            
        nrrd_files = glob.glob(os.path.join(case_dir, "*.nrrd"))
        
        for nrrd_path in nrrd_files:
            filename = os.path.basename(nrrd_path)
            
            # Extract the base name to construct the target name
            # e.g. T1_ax_label1.nrrd -> flip_T1_ax_label1.nii.gz
            # Mapping logic based on user request "T1_ax_label1.nrrd" -> "flip_T1_ax_image.nii.gz" (Wait, user said T1_ax_label1-> flip_T1_ax_image? Probably meant label1. Let's make it flexible)
            
            # The user's exact phrase: "T1_ax_label1.nrrd ... 转换为 flip_T1_ax_image.nii.gz 这种形式"
            # It's likely they meant prepending "flip_" and changing extension to .nii.gz.
            # Example: T1_ax_label1.nrrd -> flip_T1_ax_label1.nii.gz
            # Example: T2_ax_image.nrrd -> flip_T2_ax_image.nii.gz
            
            if filename.endswith(".nrrd"):
                base_name = filename[:-5] # remove .nrrd
                
                # Prepend 'flip_' if not already there to match the standard format in this project
                if not base_name.startswith("flip_"):
                    target_name = f"flip_{base_name}.nii.gz"
                else:
                    target_name = f"{base_name}.nii.gz"
                    
                target_path = os.path.join(case_dir, target_name)
                
                if not os.path.exists(target_path):
                    try:
                        img = sitk.ReadImage(nrrd_path)
                        sitk.WriteImage(img, target_path)
                        converted_count += 1
                        # print(f"Converted: {nrrd_path} -> {target_path}")
                    except Exception as e:
                        print(f"\nError converting {nrrd_path}: {e}")

    print(f"\nScan and Conversion Complete. Converted {converted_count} .nrrd files to .nii.gz.")

if __name__ == '__main__':
    TARGET_DATASET_DIR = "/mnt/home2/liuyujie/data/NPC_MRI/NPC_data_V20260102"
    find_and_convert_nrrd_to_nii(TARGET_DATASET_DIR)
