import sys
import os
from paths import Radiomics_preprocessed, Radiomics_raw
from experiment_planning.dataset_fingerprint_extractor import NPCDatasetFingerprintExtractor
from experiment_planning.generate_default_config import generate_default_config
from preprocessing.preprocessor import Preprocessor
from feature_extraction.configuration import load_radiomics_config
from tqdm import tqdm


def main():
    # 1. 定义路径
    dataset_id = "Dataset001_NPC_Primary_and_Instance_Prio_Lymphs" 
    raw_data_dir = os.path.join(Radiomics_raw, dataset_id)

    # 2. 定义文件名映射
    image_files = {
        # "T1": "flip_T1_ax_image.nii.gz",
        # "T1C": "flip_T1C_ax_image.nii.gz",
        "T2": "flip_T2_ax_image.nii.gz"
    }
    label_file = "segmentation.nii.gz"

    # --- Step 1: 实验规划 (Fingerprint & Config) ---
    print("\n[Step 1] Planning Experiment...")
    fingerprint_extractor = NPCDatasetFingerprintExtractor(
        dataset_path=raw_data_dir, dataset_id=dataset_id, target_channel="T2")
    # 如果指纹文件已存在，可以选择跳过
    fingerprint = fingerprint_extractor.run()
    
    config_path = generate_default_config(dataset_id, fingerprint, preprocessing_type="z_score")
    radiomics_config = load_radiomics_config(config_path)
    target_spacing = radiomics_config['setting']['resampledPixelSpacing']

    print(f"Target Spacing determined: {target_spacing}")

    # --- Step 2: 预处理 (N4 + Resample + Normalize) ---
    print("\n[Step 2] Running Preprocessing...")

    preprocessor = Preprocessor(target_spacing=target_spacing,
                                enable_n4=False,
                                enable_cropping=False,
                                lower_threshold=50)

    output_base_dir = os.path.join(Radiomics_preprocessed, dataset_id)

    cases = sorted(os.listdir(raw_data_dir))

    for case_id in tqdm(cases):
        case_dir = os.path.join(raw_data_dir, case_id)
        if not os.path.isdir(case_dir):
            continue

        # 构建当前病例的文件路径字典
        current_images = {}
        missing_file = False
        for mod, fname in image_files.items():
            p = os.path.join(case_dir, fname)
            if os.path.exists(p):
                current_images[mod] = p
            else:
                print(f"Missing {mod} for {case_id}, skipping...")
                missing_file = True
                break

        lbl_p = os.path.join(case_dir, label_file)
        if not os.path.exists(lbl_p):
            print(f"Missing mask for {case_id}, skipping...")
            missing_file = True

        if missing_file:
            continue

        # 执行预处理
        preprocessor.run_case(images_dict=current_images,
                              label_paths=[lbl_p],
                              output_dir=output_base_dir,
                              case_id=case_id)

    print(f"\nPreprocessing Finished! Output saved to: {output_base_dir}")
    print("Folder structure:")
    print(f"  {output_base_dir}/images/  (contains CaseID_T2.nii.gz, etc.)")
    print(f"  {output_base_dir}/labels/  (contains CaseID.nii.gz)")

if __name__ == "__main__":
    main()
