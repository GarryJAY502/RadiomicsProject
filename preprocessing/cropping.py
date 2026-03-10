import SimpleITK as sitk
import numpy as np
from typing import Tuple, List, Dict

def get_bbox_from_mask(mask: sitk.Image, inside_value: int = 1) -> Tuple[tuple, tuple]:
    """
    Calculate the Bounding Box of specific values in the mask.
    Return: (start_index, size) for RegionOfInterest
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    
    if not stats.HasLabel(inside_value):
        return (0, 0, 0), mask.GetSize()
        
    # GetBoundingBox 返回 (x_start, y_start, z_start, x_size, y_size, z_size)
    bbox = stats.GetBoundingBox(inside_value)
    start_index = bbox[:3]
    size = bbox[3:]
    
    return start_index, size

def crop_to_bbox(image: sitk.Image, start_index: tuple, size: tuple) -> sitk.Image:
    """
    Crop the image according to the given bbox
    """
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start_index)
    roi_filter.SetSize(size)
    return roi_filter.Execute(image)

def crop_image_based_on_nonzero(images: List[sitk.Image]) -> Tuple[List[sitk.Image], dict]:
    """
    联合所有模态的图像，计算非零区域的并集，然后进行裁剪。
    通常用于去除 MRI 周围的空气背景。
    
    Args:
        images: 一个病例的所有模态图像列表 [T1_img, T2_img, ...]
        
    Returns:
        cropped_images: 裁剪后的图像列表
        bbox_properties: 记录裁剪的坐标信息 (方便后续复原)
    """
    if not images:
        return [], {}

    # 1. 创建联合 Mask (Union of Non-zero)
    # 取第一张图作为基准
    ref_img = images[0]
    union_mask = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
    union_mask.CopyInformation(ref_img)
    
    for img in images:
        # 二值化：大于0的像素设为1
        binary = sitk.BinaryThreshold(img, lowerThreshold=1e-6, upperThreshold=float('inf'), insideValue=1, outsideValue=0)
        binary = sitk.Cast(binary, sitk.sitkUInt8)
        # 累加 Mask
        union_mask = sitk.Or(union_mask, binary)

    # 2. 计算 Bounding Box
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(union_mask)
    
    if not stats.HasLabel(1):
        # 全黑图像，不裁剪
        return images, {'cropped': False}

    bbox = stats.GetBoundingBox(1) # (x,y,z, w,h,d)
    start_index = bbox[:3]
    size = bbox[3:]

    # 3. 对所有图像应用相同的裁剪
    cropped_images = []
    for img in images:
        cropped_images.append(crop_to_bbox(img, start_index, size))
        
    return cropped_images, {
        'cropped': True, 
        'start_index': start_index, 
        'size': size,
        'original_size': ref_img.GetSize()
    }