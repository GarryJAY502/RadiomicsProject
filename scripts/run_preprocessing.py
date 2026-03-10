import sys
import os
import argparse
from typing import Dict, Optional
from paths import Radiomics_preprocessed, Radiomics_raw
from experiment_planning.dataset_fingerprint_extractor import NPCDatasetFingerprintExtractor
from experiment_planning.generate_default_config import generate_default_config
from preprocessing.preprocessor import Preprocessor
from feature_extraction.configuration import load_radiomics_config
from tqdm import tqdm


class NPCPreprocessPipeline:
    def __init__(self, dataset_id: str,
                 image_files: Optional[Dict[str, str]] = None,
                 label_files: Optional[list[str]] = None,
                 enable_n4: bool = False,
                 enable_cropping: bool = False,
                 enable_normalize: bool = False,
                 enable_foreground_extraction: bool = False,
                 save_debug_images: bool = False):
        self.dataset_id = dataset_id
        self.raw_data_dir = os.path.join(Radiomics_raw, dataset_id)
        self.output_base_dir = os.path.join(Radiomics_preprocessed, dataset_id)
        
        # Default NPC structure
        if image_files is None:
            self.image_files = {
                "T1": "flip_T1_ax_image.nii.gz",
                "T1C": "flip_T1C_ax_image.nii.gz",
                "T2": "flip_T2_ax_image.nii.gz"
            }
        else:
            self.image_files = image_files
            
        if label_files is None:
            self.label_files = ["segmentation.nii.gz"]
        else:
            self.label_files = label_files
            
        self.enable_n4 = enable_n4
        self.enable_cropping = enable_cropping
        self.enable_normalize = enable_normalize
        self.enable_foreground_extraction = enable_foreground_extraction
        self.save_debug_images = save_debug_images

    def run(self):
        # --- Step 1: 实验规划 (Fingerprint & Config) ---
        print(f"\n[Step 1] Planning Experiment for {self.dataset_id}...")
        fingerprint_extractor = NPCDatasetFingerprintExtractor(dataset_path=self.raw_data_dir, 
                                                               dataset_id=self.dataset_id, 
                                                               target_channel="T2", 
                                                               label_files=self.label_files)
        fingerprint = fingerprint_extractor.run()
        
        config_path = generate_default_config(self.dataset_id, 
                                              fingerprint, 
                                              preprocessing_type="z_score")
        radiomics_config = load_radiomics_config(config_path)
        target_spacing = radiomics_config['setting']['resampledPixelSpacing']

        print(f"Target Spacing determined: {target_spacing}")

        # --- Step 2: 预处理 (N4 + Resample + Normalize) ---
        print("\n[Step 2] Running Preprocessing...")

        preprocessor = Preprocessor(target_spacing=target_spacing,
                                    enable_n4=self.enable_n4,
                                    enable_cropping=self.enable_cropping,
                                    enable_normalize=self.enable_normalize,
                                    enable_foreground_extraction=self.enable_foreground_extraction,
                                    save_debug_images=self.save_debug_images)

        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")

        cases = sorted(os.listdir(self.raw_data_dir))

        for case_id in tqdm(cases, desc=f"Processing {self.dataset_id}"):
            case_dir = os.path.join(self.raw_data_dir, case_id)
            if not os.path.isdir(case_dir):
                continue

            current_images = {}
            missing_file = False
            for mod, fname in self.image_files.items():
                p = os.path.join(case_dir, fname)
                if os.path.exists(p):
                    current_images[mod] = p
                else:
                    print(f"Missing {mod} for {case_id}, skipping...")
                    missing_file = True
                    break

            lbl_paths = []
            for lbl_name in self.label_files:
                p = os.path.join(case_dir, lbl_name)
                if os.path.exists(p):
                    lbl_paths.append(p)
            
            if not lbl_paths:
                print(f"Missing all masks for {case_id}, skipping...")
                missing_file = True

            if missing_file:
                continue

            # 执行预处理
            preprocessor.run_case(images_dict=current_images,
                                  label_paths=lbl_paths,
                                  output_dir=self.output_base_dir,
                                  case_id=case_id)

        print(f"\nPreprocessing Finished! Output saved to: {self.output_base_dir}")
        print("Folder structure:")
        print(f"  {self.output_base_dir}/images/  (contains CaseID_T2.nii.gz, etc.)")
        print(f"  {self.output_base_dir}/labels/  (contains CaseID.nii.gz)")


def run_preprocessing_pipeline(dataset_id: str, label_files: Optional[list[str]] = None, save_debug_images: bool = False):
    """
    Helper function for backward compatibility and subparser mapping.
    """
    pipeline = NPCPreprocessPipeline(dataset_id=dataset_id, 
                                     label_files=label_files,
                                     enable_foreground_extraction=True,
                                     enable_normalize=True,
                                     save_debug_images=save_debug_images)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPC Preprocessing Pipeline")
    parser.add_argument("--dataset_id", type=str, default="Dataset001_NPC_Primary",
                        help="The dataset ID to preprocess")
    parser.add_argument("--label_files", nargs="+", default=["segmentation.nii.gz"],
                        help="List of label file names to process.")
    parser.add_argument("--save_debug_images", action="store_true",
                        help="If set, saves intermediate thresholding steps to temp directory.")
    args = parser.parse_args()
    
    run_preprocessing_pipeline(args.dataset_id, args.label_files, args.save_debug_images)
