import os
import numpy as np
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional
from preprocessing.cropping import crop_image_based_on_nonzero, crop_to_bbox
from preprocessing.resampling import resample_image_to_spacing, resample_label_to_spacing


class Preprocessor:
    def __init__(self, 
                 target_spacing: List[float] = None, 
                 enable_n4: bool = False, 
                 enable_cropping: bool = False, 
                 enable_normalize: bool = False,
                 enable_foreground_extraction: bool = False,
                 save_debug_images: bool = False):
        self.target_spacing = target_spacing
        self.enable_n4 = enable_n4
        self.enable_cropping = enable_cropping
        self.enable_normalize = enable_normalize
        self.enable_foreground_extraction = enable_foreground_extraction
        self.save_debug_images = save_debug_images

    def n4_bias_correction(self, image: sitk.Image) -> sitk.Image:
        """
        Perform N4 bias field correction after generating the mask using Otsu thresholding.
        """
        print("  Running N4 Bias Field Correction...")
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
        input_image = sitk.Cast(image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 25])
        output = corrector.Execute(input_image, mask_image)
        return output

    def extract_foreground(self, 
                           image: sitk.Image, 
                           closing_radius: int = 3,
                           debug_save_dir: Optional[str] = None,
                           modality_name: str = "") -> Tuple[sitk.Image, sitk.Image]:
        """
        4步精细化前景提取流程：
          Step 1: 直方图截断 (去除最低的2%灰度值)
          Step 2: Otsu 自动阈值 (在去噪图像上生成初始掩码)
          Step 3: 形态学孔洞填充 (闭运算 + BinaryFillhole)
          Step 4: 最大连通域提取 (Largest Connected Component)
        
        Args:
            image: 待处理的原始灰度影像
            closing_radius: 形态学闭运算的球形核半径
            debug_save_dir: 如果提供，且 self.save_debug_images 为 True，按步保存中间图片
            modality_name: 用于构造调试文件名的模态名称
        """
        def _save_debug(img, step_name):
            if self.save_debug_images and debug_save_dir:
                os.makedirs(debug_save_dir, exist_ok=True)
                p = os.path.join(debug_save_dir, f"{modality_name}_{step_name}.nii.gz")
                sitk.WriteImage(img, p)

        input_image = sitk.Cast(image, sitk.sitkFloat32)
        img_arr = sitk.GetArrayFromImage(input_image)

        # =======================================================
        # Step 1: 截断最低 2% 灰度值
        # =======================================================
        # 获取去重并排序后的唯一像素值
        unique_vals = np.unique(img_arr)
        # 在唯一像素值集合中计算 2% 分位数
        p2 = np.percentile(unique_vals, 2)
        # 将小于等于 2% 分位数的像素硬生生置为 0
        img_arr_clipped = np.where(img_arr <= p2, 0, img_arr)
        
        clipped_image = sitk.GetImageFromArray(img_arr_clipped)
        clipped_image.CopyInformation(input_image)
        _save_debug(clipped_image, "step1_clipped_2percent")

        # =======================================================
        # Step 2: Otsu 自动阈值分割 (基于截断后的图像)
        # =======================================================
        otsu_mask = sitk.OtsuThreshold(clipped_image, 0, 1, 256)
        _save_debug(otsu_mask, "step2_otsu_mask")

        # =======================================================
        # Step 3: 形态学闭运算 + 孔洞填充
        # =======================================================
        closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        closing_filter.SetKernelRadius(closing_radius)
        closing_filter.SetKernelType(sitk.sitkBall)
        closed_mask = closing_filter.Execute(otsu_mask)

        filled_mask = sitk.BinaryFillhole(closed_mask)
        _save_debug(filled_mask, "step3_filled_mask")

        # =======================================================
        # Step 4: 最大连通域提取 (LCC)
        # =======================================================
        # 1. ConnectedComponent 标记所有连通域
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled_mask = cc_filter.Execute(filled_mask)
        
        # 2. RelabelComponent 根据面积降序重新标记 (1 是最大的)
        relabel_filter = sitk.RelabelComponentImageFilter()
        relabel_filter.SortByObjectSizeOn()
        relabeled_mask = relabel_filter.Execute(labeled_mask)
        
        # 3. 只保留标号为 1 的最大连通域
        largest_cc_mask = relabeled_mask == 1
        _save_debug(largest_cc_mask, "step4_largest_cc_mask")

        # =======================================================
        # Step 5: 应用最终掩码 (在 Step 1 的图像上)
        # =======================================================
        foreground_mask_float = sitk.Cast(largest_cc_mask, sitk.sitkFloat32)
        masked_image = sitk.Multiply(clipped_image, foreground_mask_float)
        masked_image.CopyInformation(image)

        return masked_image, largest_cc_mask

    def z_score_normalization(self, image: sitk.Image, foreground_mask: Optional[sitk.Image] = None) -> sitk.Image:
        """
        Z-Score 归一化: (Img - Mean) / Std
        
        如果提供了前景掩码，则只在前景区域计算 mean/std。
        否则回退到使用 img > 0 作为掩码。
        """
        image = sitk.Cast(image, sitk.sitkFloat32)
        img_arr = sitk.GetArrayFromImage(image)

        if foreground_mask is not None:
            mask = sitk.GetArrayFromImage(foreground_mask).astype(bool)
        else:
            mask = img_arr > 0

        if np.any(mask):
            mean = np.mean(img_arr[mask])
            std = np.std(img_arr[mask])
            img_arr[mask] = (img_arr[mask] - mean) / (std + 1e-8)
            img_arr[~mask] = 0
        
        new_img = sitk.GetImageFromArray(img_arr)
        new_img.CopyInformation(image)
        return new_img
    
    def run_case(self, 
                images_dict: Dict[str, str],
                label_paths: List[str], 
                output_dir: str, 
                case_id: str):
        """
        Preprocesses a single case: Crop → N4 → Resample → Foreground Extraction → Normalize → Save
        处理单个病例的所有模态和多个标签
        images_dict: {'T1': 'path/to/t1', 'T2': 'path/to/t2', ...}
        label_paths: ['path/to/mask1', 'path/to/mask2', ...]
        """
        
        loaded_images = {}
        modality_order = list(images_dict.keys())
        for mod in modality_order:
            loaded_images[mod] = sitk.ReadImage(images_dict[mod])

        loaded_labels = []
        label_basenames = []
        for lp in label_paths:
            if os.path.exists(lp):
                loaded_labels.append(sitk.ReadImage(lp))
                label_basenames.append(os.path.basename(lp))

        # --- Step A: Cropping ---
        if self.enable_cropping:
            print(f"  [Cropping] Calculating bounding box based on non-zero regions...")
            img_list = [loaded_images[mod] for mod in modality_order]
            
            cropped_imgs_list, crop_info = crop_image_based_on_nonzero(img_list)
            
            if crop_info['cropped']:
                for idx, mod in enumerate(modality_order):
                    loaded_images[mod] = cropped_imgs_list[idx]
                
                print(f"  [Cropping] Applied bbox: {crop_info['start_index']} size: {crop_info['size']}")
                for i in range(len(loaded_labels)):
                    loaded_labels[i] = crop_to_bbox(loaded_labels[i], crop_info['start_index'], crop_info['size'])
            else:
                print("  [Cropping] Image was empty or full, no cropping applied.")

        # --- Step: Resample Labels ---
        if self.target_spacing:
            for i in range(len(loaded_labels)):
                loaded_labels[i] = resample_label_to_spacing(loaded_labels[i], self.target_spacing)
            
        # --- Save Labels ---
        out_lbl_dir = os.path.join(output_dir, "labels", case_id)
        os.makedirs(out_lbl_dir, exist_ok=True)
        for i, lbl in enumerate(loaded_labels):
            sitk.WriteImage(lbl, os.path.join(out_lbl_dir, label_basenames[i]))

        # --- Step B, C, D, E: N4 → Resample → Foreground Extraction → Normalize ---
        for mod in modality_order:
            img = loaded_images[mod]
            foreground_mask = None
            
            # B. N4 Bias Correction
            if self.enable_n4:
                try:
                    img = self.n4_bias_correction(img)
                except Exception as e:
                    print(f"   [Warning] N4 failed for {mod}, skipping. Error: {e}")

            # C. Resample Image
            if self.target_spacing:
                img = resample_image_to_spacing(img, self.target_spacing, interpolator=sitk.sitkBSpline3)
                
                # B-spline 插值可能会产生负值 (undershoot)，将负值截断为0
                img_arr = sitk.GetArrayFromImage(img)
                if np.any(img_arr < 0):
                    img_arr = np.clip(img_arr, a_min=0, a_max=None)
                    img_clipped = sitk.GetImageFromArray(img_arr)
                    img_clipped.CopyInformation(img)
                    img = img_clipped
                
                # Save debug image for resampling step
                if self.save_debug_images:
                    temp_dir = os.path.join(output_dir, "temp", case_id)
                    os.makedirs(temp_dir, exist_ok=True)
                    sitk.WriteImage(img, os.path.join(temp_dir, f"{case_id}_{mod}_step0_resampled.nii.gz"))

            # D. Foreground Extraction (Clip + Otsu + Fillhole + LCC)
            if self.enable_foreground_extraction:
                temp_dir = os.path.join(output_dir, "temp", case_id) if self.save_debug_images else None
                
                img, foreground_mask = self.extract_foreground(
                    img, 
                    closing_radius=10, 
                    debug_save_dir=temp_dir,
                    modality_name=mod
                )
                
                # 始终保存最终的 3D 前景掩码供后续（如 Radiomics 特征提取时如果需要的话）使用
                final_mask_dir = os.path.join(output_dir, "temp", case_id)
                os.makedirs(final_mask_dir, exist_ok=True)
                sitk.WriteImage(foreground_mask, os.path.join(final_mask_dir, f"{case_id}_{mod}_final_foreground_mask.nii.gz"))

            # E. Normalize (使用前景掩码计算统计量)
            if self.enable_normalize:
                img = self.z_score_normalization(img, foreground_mask=foreground_mask)
            
            # Save processed image
            out_img_path = os.path.join(output_dir, "images", f"{case_id}_{mod}.nii.gz")
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            sitk.WriteImage(img, out_img_path)
