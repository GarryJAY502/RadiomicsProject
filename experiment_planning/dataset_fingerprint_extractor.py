import os
import json
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from paths import Radiomics_preprocessed


class BaseDatasetFingerprintExtractor(ABC):

    def __init__(self, dataset_path: str, dataset_id: str):
        self.dataset_path = dataset_path
        self.dataset_id = dataset_id
        self.target_num_pixels = 10_000_000

    @abstractmethod
    def _collect_case_data(self) -> List[Dict[str, Any]]:
        """
        [Abstract Method] Subclasses must implement this method.
        Return a list of cases, where each element is a dictionary containing:
        {
            "case_id": "Case001",
            "images": {"T1": "/path/to/T1.nii.gz", "T2": "/path/to/T2.nii.gz"},
            "mask": "/path/to/mask.nii.gz"
        }
        """
        pass

    def _compute_geometry_statistics(self, case_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate geometric statistics (Spacing, Size)
        Based on the first channel of each case (assuming inter-modal registration)
        """
        print("  Computing geometry statistics...")
        spacings = []
        sizes = []

        for case in case_data_list:
            # Use the first channel as the geometric reference
            first_channel_path = list(case['images'].values())[0]
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(first_channel_path)
                reader.ReadImageInformation()
                spacings.append(reader.GetSpacing())
                sizes.append(reader.GetSize())
            except Exception as e:
                print(
                    f"  [Error] Geometry read failed for {case['case_id']}: {e}"
                )

        if not spacings:
            return {}

        return {
            "median_spacing": np.median(spacings, axis=0).tolist(),
            "median_size": np.median(sizes, axis=0).tolist(),
            "mean_spacing": np.mean(spacings, axis=0).tolist(),
            "mean_size": np.mean(sizes, axis=0).tolist(),
        }

    def _compute_intensity_statistics(self, case_data_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate intensity statistics (Mean, Std, Percentiles)
        Randomly sample 10 million pixels from the foreground (Mask > 0)
        """
        print("  Computing intensity statistics (sampling foreground pixels)...")

        num_cases = len(case_data_list)
        if num_cases == 0:
            return {}

        # Calculate the number of pixels to sample per case
        samples_per_case = max(1000, self.target_num_pixels // num_cases)

        # Identify all channel names
        all_channels = set()
        for c in case_data_list:
            all_channels.update(c['images'].keys())
        all_channels = sorted(list(all_channels))

        # Store all sampled pixel values { "T1": [p1, p2...], "T2": [...] }
        sampled_pixels = defaultdict(list)

        for case in case_data_list:
            mask_path = case.get('mask')
            if not mask_path or not os.path.exists(mask_path):
                print(f"    [Skip] No mask for {case['case_id']}, cannot sample foreground.")
                continue

            try:
                # Read Mask
                mask_img = sitk.ReadImage(mask_path)
                mask_arr = sitk.GetArrayFromImage(mask_img)

                # Get foreground indices
                foreground_indices = np.where(mask_arr > 0)
                num_foreground = len(foreground_indices[0])

                if num_foreground == 0:
                    continue

                # Determine the indices to sample
                if num_foreground > samples_per_case:
                    # Randomly select
                    selected_indices = np.random.choice(num_foreground, samples_per_case, replace=False)
                    # Construct indices for slicing
                    sample_coords = tuple(idx[selected_indices] for idx in foreground_indices)
                else:
                    # Select all
                    sample_coords = foreground_indices

                # Iterate through all channels of the case
                for channel in all_channels:
                    img_path = case['images'].get(channel)
                    if img_path and os.path.exists(img_path):
                        img = sitk.ReadImage(img_path)
                        img_arr = sitk.GetArrayFromImage(img)

                        # Extract pixels
                        pixels = img_arr[sample_coords]
                        sampled_pixels[channel].extend(pixels.astype(np.float32).tolist())

            except Exception as e:
                print(f"    [Error] Processing {case['case_id']} failed: {e}")

        # Calculate statistics
        stats = {}
        for channel, pixels in sampled_pixels.items():
            if not pixels:
                continue

            arr = np.array(pixels)
            stats[channel] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "percentile_00_5": float(np.percentile(arr, 0.5)),
                "percentile_99_5": float(np.percentile(arr, 99.5))
            }
            print(f"    [{channel}] Mean: {stats[channel]['mean']:.2f}, Std: {stats[channel]['std']:.2f}, 99.5%: {stats[channel]['percentile_99_5']:.2f}")

        return stats

    def _save_fingerprint(self, fingerprint: dict):
        output_file = os.path.join(Radiomics_preprocessed, self.dataset_id,
                                   "dataset_fingerprint.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(fingerprint, f, indent=4)
        print(f"Fingerprint saved to {output_file}")

    def run(self) -> Dict[str, Any]:
        """
        Scans the dataset to extract global properties.
        General execution logic: collect paths -> read metadata -> calculate statistics -> save
        """
        print(f"Starting Fingerprint Extraction for: {self.dataset_id}")
        case_data_list = self._collect_case_data()

        if not case_data_list:
            raise ValueError(f"No images found in {self.dataset_path}")

        # 1. Geometry
        geo_stats = self._compute_geometry_statistics(case_data_list)

        # 2. Intensity
        intensity_stats = self._compute_intensity_statistics(case_data_list)

        fingerprint = {
            "dataset_id": self.dataset_id,
            "n_cases": len(case_data_list),
            "modalities": list(intensity_stats.keys()),
            "geometry": geo_stats,
            "intensity_statistics": intensity_stats,
            # 为了兼容旧代码，保留顶层键
            "median_spacing": geo_stats.get("median_spacing"),
            "median_size": geo_stats.get("median_size")
        }

        self._save_fingerprint(fingerprint)
        return fingerprint


class NPCDatasetFingerprintExtractor(BaseDatasetFingerprintExtractor):
    """
    Specialized in handling hierarchical structures like: Dataset/CaseID/flip_T2_ax_image.nii.gz
    """

    def __init__(self,
                 dataset_path: str,
                 dataset_id: str,
                 target_channel: str = "T2",
                 label_files: List[str] = None):
        super().__init__(dataset_path, dataset_id)
        self.target_channel = target_channel
        
        if label_files is None:
            label_files = ["segmentation.nii.gz"]
            
        self.filename_map = {
            "images": {
                # "T1": "flip_T1_ax_image.nii.gz",
                # "T1C": "flip_T1C_ax_image.nii.gz",
                "T2": "flip_T2_ax_image.nii.gz"
            },
            "masks": label_files
        }

    def _collect_case_data(self) -> List[Dict[str, Any]]:
        cases = []
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        # 遍历第一层目录 (CaseID)
        dir_list = sorted([d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        print(f"Found {len(dir_list)} potential case folders.")

        for case_id in dir_list:
            case_dir = os.path.join(self.dataset_path, case_id)
            
            # 构建图像路径字典
            images = {}
            for mod, fname in self.filename_map["images"].items():
                p = os.path.join(case_dir, fname)
                if os.path.exists(p):
                    images[mod] = p
            
            # 构建 Mask 路径 (使用找到的第一个 mask 用于指纹提取)
            # Find the first existing mask path among the provided mask configs
            mask_p = None
            for mask_name in self.filename_map["masks"]:
                p = os.path.join(case_dir, mask_name)
                if os.path.exists(p):
                    mask_p = p
                    break
            
            # 只有当至少有一个图像且找到至少一个 mask 存在时才视为有效病例
            if images and mask_p is not None:
                cases.append({
                    "case_id": case_id,
                    "images": images,
                    "mask": mask_p
                })
        
        print(f"Collected {len(cases)} valid cases (with image and mask).")
        return cases
