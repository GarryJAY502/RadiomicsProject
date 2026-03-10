import sys
import os
import argparse
from typing import Optional, List
from paths import Radiomics_preprocessed, Radiomics_results
from feature_extraction.extractor import (
    SingleMaskNPCRadiomicsExtractor,
    MultiInstanceNPCRadiomicsExtractor,
    SeparateMasksNPCRadiomicsExtractor
)

def run_extraction(dataset_id: str, 
                   mode: str, 
                   num_threads: int, 
                   label_value: Optional[int] = 1,
                   label_files: Optional[List[str]] = None,
                   force_reextract: bool = False):
    """
    Main execution script for Radiomics Extraction routing.
    """
    
    preprocessed_dir = os.path.join(Radiomics_preprocessed, dataset_id)
    results_dir = os.path.join(Radiomics_results, dataset_id)
    config_file = os.path.join(preprocessed_dir, "radiomics_config.yaml")
    
    if not os.path.exists(preprocessed_dir):
        print(f"[Error] Preprocessed data not found: {preprocessed_dir}")
        return
    if not os.path.exists(config_file):
        print(f"[Error] Config file not found: {config_file}")
        print("Please run preprocessing & config generation first.")
        return

    os.makedirs(results_dir, exist_ok=True)

    if mode == "single":
        print(f"\n=== Running SINGLE Label Extraction (Label={label_value}) ===")
        output_file = os.path.join(results_dir, f"features_single.csv")
        
        mask_filename = label_files[0] if label_files else "segmentation.nii.gz"
        extractor = SingleMaskNPCRadiomicsExtractor(
            preprocessed_dir=preprocessed_dir,
            config_path=config_file,
            mask_filename=mask_filename,
            label_value=label_value,
            num_threads=num_threads
        )
        
    elif mode == "multi":
        print(f"\n=== Running MULTI Label Extraction (Auto-detect instances) ===")
        output_file = os.path.join(results_dir, "features_multi.csv")
        
        mask_filename = label_files[0] if label_files else "segmentation.nii.gz"
        extractor = MultiInstanceNPCRadiomicsExtractor(
            preprocessed_dir=preprocessed_dir,
            config_path=config_file,
            mask_filename=mask_filename,
            num_threads=num_threads
        )
        
    elif mode == "separate":
        print(f"\n=== Running SEPARATE Masks Extraction (Two mask files) ===")
        output_file = os.path.join(results_dir, "features_separate.csv")
        
        primary_filename = label_files[0] if label_files and len(label_files) > 0 else "segmentation_primary.nii.gz"
        lymph_filename = label_files[1] if label_files and len(label_files) > 1 else "segmentation_lymph.nii.gz"
        
        extractor = SeparateMasksNPCRadiomicsExtractor(
            preprocessed_dir=preprocessed_dir,
            config_path=config_file,
            primary_filename=primary_filename,
            lymph_filename=lymph_filename,
            num_threads=num_threads
        )
    else:
        print(f"[Error] Unknown mode: {mode}")
        return

    # Execute
    if os.path.exists(output_file) and not force_reextract:
        print(f"Output file exists: {output_file}. Use --force to overwrite.")
    else:
        df = extractor.run(output_file=output_file)
        print("Extraction flow completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Radiomics Feature Extraction Script")
    
    parser.add_argument("dataset_id", type=str, help="The Dataset ID (e.g., Dataset001_NPC_Processed)")
    
    parser.add_argument("--mode", type=str, choices=["single", "multi", "separate"], default="single",
                        help="Extraction mode: 'single', 'multi', or 'separate'. Default: single")
    
    parser.add_argument("--cores", type=int, default=8, 
                        help="Number of CPU threads/processes. Default: 8")
    
    parser.add_argument("--label", type=int, default=1, 
                        help="[Single Mode Only] The label value to extract. Default: 1")
                        
    parser.add_argument("--label_files", nargs="+", default=None, 
                        help="List of mask files to use. E.g. 'segmentation.nii.gz' or 'segmentation_primary.nii.gz segmentation_lymph.nii.gz'. Default resolves automatically based on mode.")
    
    parser.add_argument("--force", action="store_true", 
                        help="Force re-extraction even if output file exists.")

    args = parser.parse_args()
    
    run_extraction(
        dataset_id=args.dataset_id,
        mode=args.mode,
        num_threads=args.cores,
        label_value=args.label,
        label_files=args.label_files,
        force_reextract=args.force
    )