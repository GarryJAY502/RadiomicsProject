import os
import json
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from multiprocessing import Pool
import logging
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from tqdm import tqdm

logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

def run_extraction_worker(args: Tuple) -> List[Dict[str, Any]]:
    """
    Unified worker function for radiomics extraction.
    args: (case_id, modality, image_path, mask_configs, params_file)
    mask_configs is a list of dicts: [{'mask_path': str, 'label_val': int, 'roi_type': str}]
    Returns a list of extracted dictionaries for this Case+Modality.
    """
    case_id, modality, image_path, mask_configs, params_file = args
    results = []
    
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        
        # To avoid reading the same mask multiple times, group by mask_path
        mask_groups = {}
        for config in mask_configs:
            m_path = config['mask_path']
            if m_path not in mask_groups:
                mask_groups[m_path] = []
            mask_groups[m_path].append(config)
            
        for mask_path, configs in mask_groups.items():
            if not os.path.exists(mask_path):
                print(f"[Warn] Mask not found: {mask_path}")
                continue
                
            for config in configs:
                label_val = config['label_val']
                roi_type = config['roi_type']
                
                try:
                    res = extractor.execute(image_path, mask_path, label=label_val)
                    clean_res = {
                        'CaseID': case_id,
                        'ROI_Type': roi_type,
                        'Modality': modality
                    }
                    if label_val is not None:
                        clean_res['LabelID'] = label_val
                        
                    for k, v in res.items():
                        if not k.startswith("diagnostics"):
                            clean_res[f"{modality}_{k}"] = float(v)
                            
                    results.append(clean_res)
                except Exception as e:
                    print(f"[Warn] Failed {case_id} {modality} {roi_type} (Label={label_val}): {e}")
                    
    except Exception as e:
        print(f"[Error] Worker failed for {case_id} {modality}: {e}")
        
    return results


class BaseRadiomicsExtractor(ABC):
    """
    Robust base class for Radiomics Extraction.
    Handles parallel processing, temp saving, and memory merging.
    Subclasses must implement `_build_extraction_tasks()`.
    """
    def __init__(self, preprocessed_dir: str, config_path: str, num_threads: int = 4, temp_dir: str = None):
        self.preprocessed_dir = preprocessed_dir
        self.config_path = config_path
        self.num_threads = num_threads
        
        self.images_dir = os.path.join(preprocessed_dir, "images")
        self.labels_dir = os.path.join(preprocessed_dir, "labels")
        
        self.temp_dir = temp_dir if temp_dir else os.path.join(preprocessed_dir, "temp_extraction_results")
        os.makedirs(self.temp_dir, exist_ok=True)

    @abstractmethod
    def _build_extraction_tasks(self) -> List[Tuple]:
        """
        [Abstract Method] Subclasses construct tasks.
        Returns: list of (case_id, modality, image_path, mask_configs, params_file)
        mask_configs: [{'mask_path': str, 'label_val': int, 'roi_type': str}]
        """
        pass

    def run(self, output_file: str = None) -> pd.DataFrame:
        tasks = self._build_extraction_tasks()
        print(f"Collected {len(tasks)} tasks (Cases * Modalities).")
        
        if not tasks:
            print("No features extracted.")
            return pd.DataFrame()

        finished_files = set(os.listdir(self.temp_dir))
        args_to_run = []
        for args in tasks:
            case_id, modality = args[0], args[1]
            json_name = f"{case_id}_{modality}.json"
            if json_name not in finished_files:
                args_to_run.append(args)

        if len(args_to_run) < len(tasks):
            print(f"Resuming: Skipping {len(tasks) - len(args_to_run)} processed tasks.")

        # Execute Parallel Extraction
        if args_to_run:
            with Pool(self.num_threads) as p:
                iterator = p.imap_unordered(run_extraction_worker, args_to_run)
                for res_list in tqdm(iterator, total=len(args_to_run), desc="Extracting"):
                    if not res_list:
                        continue
                        
                    first_item = res_list[0]
                    case_id = first_item['CaseID']
                    modality = first_item['Modality']
                    save_path = os.path.join(self.temp_dir, f"{case_id}_{modality}.json")
                    
                    try:
                        with open(save_path, 'w') as f:
                            json.dump(res_list, f)
                    except Exception as e:
                        print(f"Error saving temp file {save_path}: {e}")

        # Merging Stage
        print("Merging temporary files...")
        final_data = []
        temp_files = sorted(os.listdir(self.temp_dir))
        
        for fname in tqdm(temp_files, desc="Merging"):
            if not fname.endswith(".json"): continue
            path = os.path.join(self.temp_dir, fname)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    final_data.extend(data)
            except Exception as e:
                print(f"Corrupted file: {fname}. Error: {e}")

        df = pd.DataFrame(final_data)
        if df.empty:
            print("No features extracted.")
            return df

        # Pivot
        print("Pivoting table to wide format (One row per ROI)...")
        pivot_df = self._pivot_to_wide_format(df)

        # Save Results
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            base_path = os.path.splitext(output_file)[0]
            
            try:
                pivot_df.to_parquet(f"{base_path}.parquet", index=False)
                print(f"Final features saved to Parquet: {base_path}.parquet")
            except ImportError:
                print("[Warning] 'pyarrow' not installed. Skipping Parquet save.")
                
            pivot_df.to_csv(f"{base_path}.csv", index=False)
            print(f"Final features saved to CSV: {base_path}.csv")

        return pivot_df

    def _pivot_to_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Long -> Wide based on Modality, merging T1, T2 etc. into a single row.
        Primary Keys: ['CaseID', 'ROI_Type', 'LabelID'] (if LabelID exists)
        """
        keys = ['CaseID', 'ROI_Type']
        if 'LabelID' in df.columns:
            keys.append('LabelID')

        if 'Modality' in df.columns:
            df = df.drop(columns=['Modality'])

        return df.groupby(keys, as_index=False).first()

    def _get_image_files(self) -> List[str]:
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        return [f for f in sorted(os.listdir(self.images_dir)) if f.endswith(".nii.gz")]

    def _parse_image_filename(self, img_name: str) -> Tuple[str, str]:
        base_name = img_name.replace(".nii.gz", "")
        parts = base_name.split('_')
        modality = parts[-1]
        case_id = "_".join(parts[:-1])
        return case_id, modality


class SingleMaskNPCRadiomicsExtractor(BaseRadiomicsExtractor):
    """
    Extracts features from a single mask file.
    Only extracts the specified label_value.
    Suitable for Primary-only or strictly Combined-as-1 setups.
    """
    def __init__(self, preprocessed_dir: str, config_path: str, mask_filename: str = "segmentation.nii.gz", label_value: int = 1, num_threads: int = 4):
        super().__init__(preprocessed_dir, config_path, num_threads)
        self.mask_filename = mask_filename
        self.label_value = label_value

    def _build_extraction_tasks(self) -> List[Tuple]:
        tasks = []
        for img_name in self._get_image_files():
            case_id, modality = self._parse_image_filename(img_name)
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.labels_dir, case_id, self.mask_filename)

            if os.path.exists(mask_path):
                mask_configs = [{'mask_path': mask_path, 'label_val': self.label_value, 'roi_type': 'primary'}]
                tasks.append((case_id, modality, img_path, mask_configs, self.config_path))
                
        return tasks


class MultiInstanceNPCRadiomicsExtractor(BaseRadiomicsExtractor):
    """
    Extracts features from an entire unified mask file continuously.
    Automatically identifies integers (1, 2, 3...) and extracts them all.
    Suitable for Combined Instance masking.
    """
    def __init__(self, preprocessed_dir: str, config_path: str, mask_filename: str = "segmentation.nii.gz", num_threads: int = 4):
        super().__init__(preprocessed_dir, config_path, num_threads)
        self.mask_filename = mask_filename

    def _build_extraction_tasks(self) -> List[Tuple]:
        tasks = []
        for img_name in self._get_image_files():
            case_id, modality = self._parse_image_filename(img_name)
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.labels_dir, case_id, self.mask_filename)

            if os.path.exists(mask_path):
                try:
                    mask_obj = sitk.ReadImage(mask_path)
                    mask_arr = sitk.GetArrayFromImage(mask_obj)
                    unique_labels = [int(l) for l in np.unique(mask_arr) if l > 0]
                    
                    mask_configs = []
                    for lbl in unique_labels:
                        roi_type = "primary" if lbl == 1 else f"lymph_{lbl}"
                        mask_configs.append({'mask_path': mask_path, 'label_val': lbl, 'roi_type': roi_type})
                        
                    if mask_configs:
                        tasks.append((case_id, modality, img_path, mask_configs, self.config_path))
                except Exception as e:
                    print(f"[Error] Failed to read mask {mask_path}: {e}")
                    
        return tasks


class SeparateMasksNPCRadiomicsExtractor(BaseRadiomicsExtractor):
    """
    Extracts features from two entirely distinct spatial mask files.
    e.g. segmentation_primary.nii.gz and segmentation_lymph.nii.gz
    """
    def __init__(self, preprocessed_dir: str, config_path: str, 
                 primary_filename: str = "segmentation_primary.nii.gz",
                 lymph_filename: str = "segmentation_lymph.nii.gz",
                 num_threads: int = 4):
        super().__init__(preprocessed_dir, config_path, num_threads)
        self.primary_filename = primary_filename
        self.lymph_filename = lymph_filename

    def _build_extraction_tasks(self) -> List[Tuple]:
        tasks = []
        for img_name in self._get_image_files():
            case_id, modality = self._parse_image_filename(img_name)
            img_path = os.path.join(self.images_dir, img_name)
            
            p_mask_path = os.path.join(self.labels_dir, case_id, self.primary_filename)
            l_mask_path = os.path.join(self.labels_dir, case_id, self.lymph_filename)
            
            mask_configs = []
            
            if os.path.exists(p_mask_path):
                mask_configs.append({'mask_path': p_mask_path, 'label_val': 1, 'roi_type': 'primary'})
                
            if os.path.exists(l_mask_path):
                try:
                    l_mask_obj = sitk.ReadImage(l_mask_path)
                    l_mask_arr = sitk.GetArrayFromImage(l_mask_obj)
                    l_labels = [int(l) for l in np.unique(l_mask_arr) if l > 0]
                    for lbl in l_labels:
                        mask_configs.append({'mask_path': l_mask_path, 'label_val': lbl, 'roi_type': f"lymph_{lbl}"})
                except Exception as e:
                    print(f"[Error] Failed to read lymph mask {l_mask_path}: {e}")
            
            if mask_configs:
                tasks.append((case_id, modality, img_path, mask_configs, self.config_path))
                
        return tasks
