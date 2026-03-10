import SimpleITK as sitk
import numpy as np
from typing import List, Tuple

def resample_image_to_spacing(image: sitk.Image, new_spacing: List[float], interpolator=sitk.sitkBSpline3) -> sitk.Image:
    """
    Resamples an image to a new spacing.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(image)

def resample_label_to_spacing(label: sitk.Image, new_spacing: List[float]) -> sitk.Image:
    """
    Resamples a label (mask) using Nearest Neighbor interpolation.
    """
    return resample_image_to_spacing(label, new_spacing, interpolator=sitk.sitkNearestNeighbor)
