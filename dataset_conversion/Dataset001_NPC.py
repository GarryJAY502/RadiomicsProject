import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


class GeometryMismatchError(Exception):
    """
    Exception raised when spatial geometries (Size, Spacing, Origin, Direction) 
    between two images or an image and its label do not match.
    """
    pass


class NPCDatasetConverter:

    def __init__(self,
                 raw_data_dir: str,
                 output_data_dir: str,
                 reference_channel: str = "T2"):
        self.raw_dir = raw_data_dir
        self.out_root = output_data_dir
        self.ref_channel = reference_channel
        self.log_messages = []

    def _log(self, msg: str):
        tqdm.write(msg)
        self.log_messages.append(msg)

    def _read_image(self, path: str) -> sitk.Image:
        return sitk.ReadImage(path)

    def _check_geometry_match(self, image_a: sitk.Image, image_b: sitk.Image, tolerance: float = 1e-4) -> None:
        """
        Strictly checks if Size, Spacing, Origin, and Direction match between two images.
        Raises GeometryMismatchError if they differ.
        """
        if image_a.GetSize() != image_b.GetSize():
            raise GeometryMismatchError(f"Size mismatch: {image_a.GetSize()} vs {image_b.GetSize()}")
        
        if not np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=tolerance):
            raise GeometryMismatchError(f"Spacing mismatch: {image_a.GetSpacing()} vs {image_b.GetSpacing()}")
            
        if not np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=tolerance):
            raise GeometryMismatchError("Origin mismatch")
            
        if not np.allclose(image_a.GetDirection(), image_b.GetDirection(), atol=tolerance):
            raise GeometryMismatchError("Direction mismatch")

    def _align_image_geometry(self,
                              image_target: sitk.Image,
                              image_reference: sitk.Image,
                              is_mask: bool = False) -> sitk.Image:
        try:
            self._check_geometry_match(image_reference, image_target)
            return image_target
        except GeometryMismatchError:
            pass # Needs resampling

        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline3
        default_value = 0 if is_mask else 0.0

        resampled_target = sitk.Resample(image1=image_target,
                                         referenceImage=image_reference,
                                         transform=sitk.Transform(),
                                         interpolator=interpolator,
                                         defaultPixelValue=default_value)

        if is_mask and resampled_target.GetPixelID() != image_target.GetPixelID():
            resampled_target = sitk.Cast(resampled_target, image_target.GetPixelID())

        return resampled_target

    def _load_and_clean_mask(self,
                             mask_path: str,
                             image_reference: sitk.Image,
                             target_val: Optional[int] = None,
                             is_instance: bool = False) -> Optional[np.ndarray]:
        if not os.path.exists(mask_path):
            return None

        mask = self._read_image(mask_path)

        # Basic Z-axis check first for quick fail
        if mask.GetSize()[2] != image_reference.GetSize()[2]:
            raise GeometryMismatchError(
                f"Mask Z-depth {mask.GetSize()[2]} != Image Z-depth {image_reference.GetSize()[2]}"
            )

        mask_aligned = self._align_image_geometry(mask, image_reference, is_mask=True)
        arr = sitk.GetArrayFromImage(mask_aligned)
        
        # Cast to int32 first to avoid numpy zeros_like issues with float64 arrays
        arr = np.round(arr).astype(np.int32)
        arr[arr < 0] = 0

        if is_instance:
            return arr.astype(np.uint16)
        elif target_val is not None:
            arr_new = np.zeros_like(arr, dtype=np.uint16)
            arr_new[arr > 0] = target_val
            return arr_new

        return arr.astype(np.uint16)

    def _merge_masks(self, 
                     arr_primary: Optional[np.ndarray], 
                     arr_lymph: Optional[np.ndarray], 
                     priority: str, 
                     is_instance_lymph: bool = False) -> Optional[np.ndarray]:
        """
        Merge primary and lymph arrays based on the given priority.
        If is_instance_lymph is True, the algorithm shifts the instance IDs to avoid conflict with Primary=1.
        """
        if arr_primary is None and arr_lymph is None:
            return None
        if arr_primary is None:
            return arr_lymph
        if arr_lymph is None:
            return arr_primary

        final_mask = np.zeros_like(arr_primary)
        
        shifted_lymph = arr_lymph.copy()
        lymph_mask = arr_lymph > 0
        
        # If instance mode, we map Lymph IDs (1, 2, 3...) to (2, 3, 4...) because 1 is reserved for Primary
        if is_instance_lymph:
            shifted_lymph[lymph_mask] = arr_lymph[lymph_mask] + 1
        else:
            # Force lymph mask to be 2
            shifted_lymph[lymph_mask] = 2

        if priority == 'lymph_first':
            # Lymph covers Overlap
            final_mask[arr_primary == 1] = 1
            final_mask[lymph_mask] = shifted_lymph[lymph_mask]
        else: # primary_first
            # Primary covers Overlap
            final_mask[lymph_mask] = shifted_lymph[lymph_mask]
            final_mask[arr_primary == 1] = 1

        return final_mask

    def _check_spatial_consistency(self, mask1: sitk.Image, mask2: sitk.Image) -> bool:
        """
        Used when saving two masks separately to ensure they are perfectly aligned.
        """
        try:
            self._check_geometry_match(mask1, mask2)
            return True
        except GeometryMismatchError as e:
            self._log(f"[Spatial Consistency Failed] {e}")
            return False

    def build_task(self, task_config: dict, case_list: List[str], channel_mapping: dict):
        task_name = task_config['name']
        task_out_dir = os.path.join(self.out_root, task_name)
        os.makedirs(task_out_dir, exist_ok=True)

        target_mode = task_config['target_mode']
        priority = task_config.get('overlap_priority', 'primary_first')
        is_instance_lymph = task_config.get('is_instance_lymph', False)

        self._log(f"\n=== Building Task: {task_name} ===")
        self._log(f"Mode: {target_mode} | Overlap Priority: {priority} | Instance: {is_instance_lymph}")

        skipped_channel_count = 0
        geometry_mismatch_count = 0

        for case_id in tqdm(case_list, desc=task_name):
            case_raw_dir = os.path.join(self.raw_dir, case_id)
            case_out_dir = os.path.join(task_out_dir, case_id)

            if self.ref_channel not in channel_mapping: 
                continue
            
            ref_path = os.path.join(case_raw_dir, channel_mapping[self.ref_channel])
            if not os.path.exists(ref_path):
                continue

            ref_img = self._read_image(ref_path)
            verified_case = True
            channel_images = {self.ref_channel: ref_img}

            # 1. Process and Verify Image Channels
            for ch, fname in channel_mapping.items():
                if ch == self.ref_channel:
                    continue
                p = os.path.join(case_raw_dir, fname)
                if os.path.exists(p):
                    img = self._read_image(p)
                    # We strictly enforce Z-axis match for images
                    if img.GetSize()[2] != ref_img.GetSize()[2]:
                        verified_case = False
                        break
                    channel_images[ch] = self._align_image_geometry(img, ref_img, is_mask=False)

            if not verified_case:
                skipped_channel_count += 1
                continue

            # 2. Process Labels
            try:
                p_file = task_config['files'].get('primary')
                l_file = task_config['files'].get('lymph')
                
                arr_p, arr_l = None, None
                
                if p_file:
                    fpath = os.path.join(case_raw_dir, p_file)
                    arr_p = self._load_and_clean_mask(fpath, ref_img, target_val=1)
                
                if l_file:
                    fpath = os.path.join(case_raw_dir, l_file)
                    arr_l = self._load_and_clean_mask(fpath, ref_img, is_instance=is_instance_lymph, target_val=2 if not is_instance_lymph else None)

                # 严格模式检查：只要配置中规定了该掩码文件，则该掩码文件必须存在缺失时直接跳过
                if p_file and arr_p is None:
                    continue
                if l_file and arr_l is None:
                    continue

                output_masks = {} # Dict of {filename: mask_array}

                if target_mode == 'primary_only':
                    if arr_p is not None:
                        output_masks["segmentation_primary.nii.gz"] = arr_p
                
                elif target_mode == 'lymph_only':
                    if arr_l is not None:
                        # Output as 1 if it wasn't instance
                        if not is_instance_lymph:
                            arr_l[arr_l == 2] = 1
                        output_masks["segmentation_lymph.nii.gz"] = arr_l
                
                elif target_mode == 'combined':
                    arr_merged = self._merge_masks(arr_p, arr_l, priority, is_instance_lymph)
                    if arr_merged is not None:
                        output_masks["segmentation.nii.gz"] = arr_merged

                elif target_mode == 'separate' or target_mode == 'separate_instance':
                    if arr_p is not None and arr_l is not None:
                        # In separate mode, we verify they spatially align perfectly.
                        img_p = sitk.GetImageFromArray(arr_p)
                        img_p.CopyInformation(ref_img)
                        img_l = sitk.GetImageFromArray(arr_l)
                        img_l.CopyInformation(ref_img)
                        
                        if self._check_spatial_consistency(img_p, img_l):
                           output_masks["segmentation_primary.nii.gz"] = arr_p
                           output_masks["segmentation_lymph.nii.gz"] = arr_l
                        else:
                            raise GeometryMismatchError("Primary and Lymph masks are not spatially consistent.")

            except GeometryMismatchError as e:
                self._log(f"[Drop] {case_id}: {e}")
                geometry_mismatch_count += 1
                continue
            except Exception as e:
                self._log(f"[Error] {case_id}: {e}")
                continue

            # 3. Save Results
            if output_masks:
                os.makedirs(case_out_dir, exist_ok=True)
                # Ensure output array is not empty
                if not any(arr.any() for arr in output_masks.values()):
                    self._log(f"[Skip] {case_id}: No valid label content.")
                    continue

                for ch, img_obj in channel_images.items():
                    sitk.WriteImage(img_obj, os.path.join(case_out_dir, channel_mapping[ch]))
                
                for mask_name, mask_arr in output_masks.items():
                    final_mask_img = sitk.GetImageFromArray(mask_arr)
                    final_mask_img.CopyInformation(ref_img)
                    sitk.WriteImage(final_mask_img, os.path.join(case_out_dir, mask_name))
            else:
                self._log(f"[Skip] {case_id}: Required labels missing.")

        self._log(
            f"Task {task_name} Finished. Image Drop: {skipped_channel_count}, Geometry Drop: {geometry_mismatch_count}"
        )


if __name__ == '__main__':
    # ================= General Config =================
    RAW_DIR = "/mnt/home2/liuyujie/data/NPC_MRI/NPC_data_V20260102/"
    OUTPUT_ROOT = "/mnt/home2/liuyujie/data/NPC_MRI/Radiomics_raw/"

    CHANNELS = {
        "T2": "flip_T2_ax_image.nii.gz",  # Used as reference normally
        # "T1": "flip_T1_ax_image.nii.gz",
        # "T1C": "flip_T1C_ax_image.nii.gz"
    }

    # ================= Task Configurations =================
    TASK_LIST = [
        # # --- Task 1: Primary Only ---
        # {
        #     "name": "Dataset001_NPC_Primary",
        #     "target_mode": "primary_only", 
        #     "files": {
        #         "primary": "flip_T2_ax_label1.nii.gz", 
        #         "lymph": None
        #         }
        # },

        # # --- Task 2: Standard Combined (Primary and Global Lymph) ---
        # {
        #     "name": "Dataset002_NPC_Combined_Prio_Primary",
        #     "target_mode": "combined",
        #     "overlap_priority": "primary_first", 
        #     "is_instance_lymph": False,
        #     "files": {
        #         "primary": "flip_T2_ax_label1.nii.gz", 
        #         "lymph": "flip_T2_ax_label2.nii.gz"
        #     }
        # },

        # # --- Task 3: Instance Lymph Combined (Lymph Instance Overrides Primary) ---
        # {
        #     "name": "Dataset003_NPC_Combined_Instance_Prio_Lymphs",
        #     "target_mode": "combined",
        #     "overlap_priority": "lymph_first",
        #     "is_instance_lymph": True,
        #     "files": {
        #         "primary": "flip_T2_ax_label1.nii.gz", 
        #         "lymph": "flip_T2_ax_lymph_instance.nii.gz"
        #     }
        # },

        # # --- Task 4: Separate Labels with Consistency Checks ---
        # {
        #     "name": "Dataset004_NPC_Separate",
        #     "target_mode": "separate",
        #     "is_instance_lymph": False,
        #     "files": {
        #         "primary": "flip_T2_ax_label1.nii.gz", 
        #         "lymph": "flip_T2_ax_label2.nii.gz"
        #     }
        # },

        # --- Task 5: Separate Labels with Instance Consistency Checks ---
        {
            "name": "Dataset005_NPC_Separate",
            "target_mode": "separate_instance",
            "is_instance_lymph": True,
            "files": {
                "primary": "flip_T2_ax_label1.nii.gz", 
                "lymph": "flip_T2_ax_lymph_instance.nii.gz"
            }
        },
        # --- Task 6: Separate Labels with Instance Consistency Checks didnot normalize ---
        {
            "name": "Dataset006_NPC_Separate_not_normalize",
            "target_mode": "separate_instance",
            "is_instance_lymph": True,
            "files": {
                "primary": "flip_T2_ax_label1.nii.gz", 
                "lymph": "flip_T2_ax_lymph_instance.nii.gz"
            }
        }
    ]

    # ================= Execution =================
    converter = NPCDatasetConverter(raw_data_dir=RAW_DIR, output_data_dir=OUTPUT_ROOT, reference_channel="T2")
    
    if os.path.exists(RAW_DIR):
        cases = sorted(os.listdir(RAW_DIR))
        for task in TASK_LIST:
            converter.build_task(task, cases, CHANNELS)
    else:
        print(f"Error: Raw directory '{RAW_DIR}' does not exist.")
