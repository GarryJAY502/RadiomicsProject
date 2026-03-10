import os
import argparse
from scripts.run_feature_extraction import run_extraction
from scripts.run_preprocessing import run_preprocessing_pipeline
from scripts.verify_NPC_preprocessing import main as run_npc_verify
from scripts.NPC_verify_raw_dataset_integrity import main as run_raw_integrity_check
from utils.logger import setup_global_logging

if __name__ == "__main__":
    # Initialize global logging before anything else
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    setup_global_logging(log_dir=log_dir, task_name="run_pipeline")
    
    parser = argparse.ArgumentParser(description="Unified Pipeline and CLI for Radiomics Project")
    subparsers = parser.add_subparsers(dest="task", required=True, help="Task to run")
    
    # 2. Extract command
    parser_extract = subparsers.add_parser("extract", help="Run feature extraction only")
    parser_extract.add_argument("dataset_id", type=str, help="Dataset ID")
    parser_extract.add_argument("--mode", type=str, choices=["single", "multi", "separate"], default="single",
                                help="Extraction mode: 'single', 'multi', or 'separate'. Default: single")
    parser_extract.add_argument("--cores", type=int, default=8, help="Number of CPU cores. Default: 8")
    parser_extract.add_argument("--label", type=int, default=1, help="[Single Mode] Label to extract. Default: 1")
    parser_extract.add_argument("--label_files", nargs="+", default=None, help="List of mask files to use.")
    parser_extract.add_argument("--force", action="store_true", help="Force re-extraction")
    
    # 3. NPC Preprocessing command
    parser_npc_prep = subparsers.add_parser("npc_preprocess", help="Run NPC preprocessing")
    parser_npc_prep.add_argument("dataset_id", type=str, help="Dataset ID")
    parser_npc_prep.add_argument("--label_files", nargs="+", default=None, help="List of label filenames to preprocess. Example: segmentation.nii.gz or segmentation_primary.nii.gz segmentation_lymph.nii.gz")
    parser_npc_prep.add_argument("--save_debug_images", action="store_true", help="Save intermediate foreground extraction steps for debugging")
    
    # 4. NPC Verify Preprocessing command
    parser_npc_verify = subparsers.add_parser("npc_verify_preprocess", help="Verify NPC preprocessing")
    
    # 5. Raw Dataset Integrity command
    parser_integrity = subparsers.add_parser("raw_integrity", help="Verify raw dataset integrity")
    
    args = parser.parse_args()
    
    if args.task == "extract":
        run_extraction(dataset_id=args.dataset_id,
                       mode=args.mode,
                       num_threads=args.cores,
                       label_value=args.label,
                       label_files=getattr(args, 'label_files', None),
                       force_reextract=args.force)
    elif args.task == "npc_preprocess":
        run_preprocessing_pipeline(dataset_id=args.dataset_id, 
                                   label_files=args.label_files,
                                   save_debug_images=args.save_debug_images)
    elif args.task == "npc_verify_preprocess":
        run_npc_verify()
    elif args.task == "raw_integrity":
        run_raw_integrity_check()
