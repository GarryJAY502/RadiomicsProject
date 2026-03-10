import json
import os
from datetime import datetime
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from typing import List, Dict, Optional


def get_sitk_properties(img: sitk.Image) -> dict:
    """
        Extract the key geometric properties of SimpleITK images
    """
    return {
        "size": img.GetSize(),
        "origin": img.GetOrigin(),
        "spacing": img.GetSpacing(),
        "direction": img.GetDirection()
    }


def check_properties_match(property_a: dict,
                           property_b: dict,
                           name_a: str,
                           name_b: str,
                           tolerance: float = 1.e-8) -> List[str]:
    """
        Compare the consistency of geometric information between images and labels
    """
    errors = []

    # 1. check Shape (Size)
    if property_a['size'] != property_b['size']:
        errors.append(
            f"Size mismatch: {name_a} {property_a['size']} vs {name_b} {property_b['size']}"
        )
    # 2. check Spacing
    if not np.allclose(
        property_a['spacing'], property_b['spacing'], atol=tolerance):
        errors.append(
            f"Spacing mismatch: {name_a} {property_a['spacing']} vs {name_b} {property_b['spacing']}"
        )
    # 3. check Origin
    if not np.allclose(
        property_a['origin'], property_b['origin'], atol=tolerance):
        errors.append(
            f"Origin mismatch: {name_a} {property_a['origin']} vs {name_b} {property_b['origin']}"
        )
    # 4. check Direction
    if not np.allclose(
        property_a['direction'], property_b['direction'], atol=tolerance):
        errors.append(
            f"Direction mismatch: {name_a} {property_a['direction']} vs {name_b} {property_b['direction']}"
        )

    return errors


def save_reports(report_data: dict, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_folder, f"report_{timestamp}.json")
    txt_path = os.path.join(output_folder, f"report_{timestamp}.txt")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Dataset Report ===\n")
        f.write(f"Time: {report_data['meta']['timestamp']}\n")
        f.write(f"Total Checked: {report_data['meta']['total_checked']}\n")
        f.write(f"Valid: {report_data['meta']['valid_count']}\n")
        f.write(f"Failed: {report_data['meta']['failed_count']}\n")
        f.write("=" * 50 + "\n\n")

        if not report_data['errors']:
            f.write("Great! No errors found.\n")
        else:
            for case_id, error_list in report_data['errors'].items():
                f.write(f"CASE: {case_id}\n")
                for err in error_list:
                    f.write(f"  - [{err['type']}] {err['detail']}\n")
                f.write("-" * 30 + "\n")

    print(f"\n[Report Saved]\nJSON: {json_path}\nTXT:  {txt_path}")


def check_case(case_path: str, case_id: str,
               image_channel_files: Dict[str, str], mask_files: Dict[str, str],
               expected_labels: List[int]) -> List[Dict[str, str]]:
    """
    check a single case
    :param channel_files: 模态名称到文件名的映射, e.g., {'T1': 't1.nii.gz', 'T2': 't2.nii.gz'}
    :param label_files: 模态对应的标签文件映射, e.g., {'T1': 'label_t1.nii.gz', 'T2': 'label_t2.nii.gz'}
                        或者如果只有一个通用标签: {'General': 'mask.nii.gz'}
    检查单个病例，返回错误列表。如果列表为空，说明检查通过。
    返回格式: [{"type": "Geometry", "detail": "..."}]
    """
    error_log = []

    # Cache the image objects for subsequent cross-comparison
    loaded_images = {}  # type: Dict[str, sitk.Image]
    loaded_masks = {}  # type: Dict[str, sitk.Image]

    # check of basic files existed
    for channel, fname in image_channel_files.items():
        image = os.path.join(case_path, fname)
        if not os.path.exists(image):
            error_log.append({
                "type": "Missing Image File",
                "detail": f"Missing channel {channel} ({fname})"
            })
            return error_log
        loaded_images[channel] = sitk.ReadImage(image)

    for channel, fname in mask_files.items():
        mask = os.path.join(case_path, fname)
        if not os.path.exists(mask):
            error_log.append({
                "type": "Missing Mask File",
                "detail": f"Missing label {channel} ({fname})"
            })
            return error_log
        loaded_masks[channel] = sitk.ReadImage(mask)

    if not loaded_images:
        error_log.append({
            "type": "Config Error",
            "detail": "No images defined to check."
        })
        return error_log

    # Image-to-Image Consistency Check (Are multi-modal images aligned with each other?)
    image_keys = list(loaded_images.keys())
    reference_image_key = image_keys[0]
    reference_image_properties = get_sitk_properties(
        loaded_images[reference_image_key])

    if len(image_keys) > 1:
        for other_key in image_keys[1:]:
            other_properties = get_sitk_properties(loaded_images[other_key])
            geo_errors = check_properties_match(
                reference_image_properties, other_properties,
                f"Image({reference_image_key})", f"Image({other_key})")
            for err in geo_errors:
                error_log.append({"type": "Image Miss Match", "detail": err})

    # 2. Image-to-Label Geometry Check (图像和标签是否对齐)
    for label_key, label_obj in loaded_masks.items():
        label_props = get_sitk_properties(label_obj)

        # 检查几何对齐
        geo_errors = check_properties_match(reference_image_properties,
                                            label_props,
                                            f"Image({reference_image_key})",
                                            f"Label({label_key})")
        for err in geo_errors:
            error_log.append({"type": "Image-Label Mismatch", "detail": err})

        # -------------------------------------------------
        # 3. Label Value Validation (标签数值合法性检查)
        # -------------------------------------------------
        label_array = sitk.GetArrayFromImage(label_obj)
        unique_values = np.unique(label_array)
        unexpected = [v for v in unique_values if v not in expected_labels]

        if unexpected:
            error_log.append({
                "type":
                "Label Value Error",
                "detail":
                f"'{label_key}' has unexpected values {unexpected}"
            })
        if len(unique_values) == 1 and unique_values[0] == 0:
            error_log.append({
                "type": "Empty Label",
                "detail": f"'{label_key}' is all 0"
            })

    # -------------------------------------------------
    # 4. Label-to-Label Consistency Check (多标签之间的一致性)
    # -------------------------------------------------
    # 如果存在多个标签文件，检查它们是否完全一致（几何+内容）
    if len(loaded_masks) > 1:
        label_keys = list(loaded_masks.keys())
        base_key = label_keys[0]
        base_arr = sitk.GetArrayFromImage(loaded_masks[base_key])
        base_props = get_sitk_properties(loaded_masks[base_key])

        for k in label_keys[1:]:
            comp_obj = loaded_masks[k]
            comp_props = get_sitk_properties(comp_obj)

            # A. 检查标签之间的几何
            geo_errors = check_properties_match(base_props, comp_props,
                                                f"Label({base_key})",
                                                f"Label({k})")

            if geo_errors:
                for err in geo_errors:
                    error_log.append({
                        "type": "Label-Label Geo Mismatch",
                        "detail": err
                    })
            else:
                # 像素比较 (仅当几何一致时)
                comp_arr = sitk.GetArrayFromImage(comp_obj)
                diff_pixels = np.sum(base_arr != comp_arr)
                if diff_pixels > 0:
                    error_log.append({
                        "type":
                        "Label Content Mismatch",
                        "detail":
                        f"'{base_key}' vs '{k}': {int(diff_pixels)} pixels differ"
                    })

    return error_log


def run_dataset_check(root_dir: str, output_dir: str,
                      channel_mapping: Dict[str, str],
                      label_mapping: Dict[str,
                                          str], expected_labels: List[int]):
    """
    主入口函数
    """
    cases = sorted(os.listdir(root_dir))
    print(f"Starting verification for {len(cases)} cases in {root_dir}...")
    print(f"Checking Channels: {list(channel_mapping.keys())}")
    print(f"Checking Labels: {list(label_mapping.keys())}")
    print(f"Expected Label Values: {expected_labels}\n")
    print(f"Report will be saved to: {output_dir}\n")

    report_data = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": root_dir,
            "total_checked": 0,
            "valid_count": 0,
            "failed_count": 0
        },
        "errors": {}  # 结构: {"CaseID": [ErrorDict, ...]}
    }

    valid_count = 0
    error_cases = []

    for case_id in tqdm(cases):
        case_path = os.path.join(root_dir, case_id)
        if not os.path.isdir(case_path):
            continue

        report_data["meta"]["total_checked"] += 1

        case_errors = check_case(case_path, case_id, channel_mapping,
                                 label_mapping, expected_labels)

        if not case_errors:
            report_data["meta"]["valid_count"] += 1
        else:
            report_data["meta"]["failed_count"] += 1
            report_data["errors"][case_id] = case_errors

    print("\n" + "=" * 50)
    print(
        f"Checked: {report_data['meta']['total_checked']} | Valid: {report_data['meta']['valid_count']} | Failed: {report_data['meta']['failed_count']}"
    )
    if report_data['meta']['failed_count'] > 0:
        print(f"Found issues in {len(report_data['errors'])} cases.")
    print("=" * 50)

    save_reports(report_data, output_dir)

    print("\n" + "=" * 50)
    print("Verification Summary")
    print("=" * 50)
    print(f"Total Cases Checked: {len(cases)}")
    print(f"Valid Cases:         {valid_count}")
    print(f"Invalid Cases:       {len(error_cases)}")
    if error_cases:
        print(f"List of Invalid Cases: {error_cases}")
    print("=" * 50)


def main():
    # ================= 配置区域 =================

    # 1. 你的原始数据根目录 (包含各个病例文件夹的目录)
    #    结构假设: /.../Data/Case001/T1.nii, /.../Data/Case001/label.nii ...
    DATA_ROOT = "/mnt/home2/liuyujie/data/NPC_MRI/NPC_data_V20260102/"
    REPORT_DIR = "/mnt/home2/liuyujie/data/NPC_MRI/NPC_data_V20260102/"

    # 2. 定义文件名映射
    #    Key是显示的名称，Value是实际的文件名
    CHANNELS_MAP = {
        "T2": "flip_T2_ax_image.nii.gz",
        "T1": "flip_T1_ax_image.nii.gz",
        "T1C": "flip_T1C_ax_image.nii.gz"
    }

    # 3. 定义标签映射 (此处实现你的第4点需求)
    #    场景 A: 你只有一个标签文件，想检查它是否合规。
    # LABELS_MAP = { "Primary": "flip_T2_ax_label1.nii.gz" }

    #    场景 B: 你有两个标签文件（例如T1和T2分开画的），想检查它们是否完全一致。
    LABELS_MAP = {
        "T2_Label": "flip_T2_ax_label2.nii.gz",
        # "T1_Label": "flip_T1_ax_label2.nii.gz", 
        # "T1C_Label": "flip_T1C_ax_label2.nii.gz",
        # "T2_Instance_Label": "flip_T2_ax_lymph_instance.nii.gz"
    }

    # 4. 预期的标签值 (包括背景0)
    #    例如: [0, 1] 表示只有背景和肿瘤; [0, 1, 2] 表示背景、肿瘤、淋巴结
    EXPECTED_LABEL_VALUES = [0, 2]

    # ================= 开始运行 =================
    run_dataset_check(DATA_ROOT, REPORT_DIR, CHANNELS_MAP, LABELS_MAP,
                      EXPECTED_LABEL_VALUES)

if __name__ == '__main__':
    main()
