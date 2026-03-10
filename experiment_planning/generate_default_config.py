import os
import yaml
from typing import Dict, Any, Type, Optional
from paths import Radiomics_preprocessed

# ==========================================
# 1. 策略基类 (Base Strategy)
# ==========================================
class RadiomicsConfigStrategy:
    """
    [扩展点] 定义如何根据预处理方式生成 PyRadiomics 配置的接口。
    如果你有新的预处理方法，请继承此类并实现相应方法。
    """
    def __init__(self, **kwargs):
        pass

    def get_setting(self) -> Dict[str, Any]:
        """返回通用的 setting 配置"""
        return {
            "interpolator": "sitkBSpline",
            "resampledPixelSpacing": None,  # 默认假设预处理已完成重采样
            "force2D": False
        }
    
    def get_bin_width(self) -> float:
        """返回推荐的 binWidth"""
        return 25.0  # 默认值

    def get_normalize(self) -> bool:
        """返回是否需要 PyRadiomics 再次进行 normalize"""
        return False
        
    def get_description(self) -> str:
        return "Base Strategy"

# ==========================================
# 2. 具体策略实现 (Concrete Strategies)
# ==========================================

class ZScoreNormalizationStrategy(RadiomicsConfigStrategy):
    """
    策略：预处理阶段已做 Z-Score (Mean=0, Std=1)。
    适用场景：MRI 常规预处理。
    """
    def get_bin_width(self) -> float:
        # Z-Score 后数值在 [-3, 3] 左右，binWidth 取 0.5 (即半个标准差) 是标准做法
        return 0.5 

    def get_normalize(self) -> bool:
        # 预处理已经做过了，这里必须关掉，否则会破坏 Z-Score 的分布
        return False
    
    def get_description(self) -> str:
        return "Z-Score (Mean=0, Std=1) -> binWidth=0.5, normalize=False"

class RawIntensityStrategy(RadiomicsConfigStrategy):
    """
    策略：输入是原始强度 (如 CT 的 HU，或未归一化的 MRI)。
    适用场景：CT 数据，或者通过 PyRadiomics 内部做归一化的 MRI。
    """
    def get_setting(self) -> Dict[str, Any]:
        settings = super().get_setting()
        # 让 PyRadiomics 内部把数据缩放到 Scale=100 (Mean*Scale/Std)
        settings["normalizeScale"] = 100
        return settings

    def get_bin_width(self) -> float:
        # 缩放后通常在 0-500 范围，binWidth=5 比较合适
        # 如果是纯 CT (未缩放)，binWidth=25
        return 5.0 

    def get_normalize(self) -> bool:
        return True
        
    def get_description(self) -> str:
        return "Raw Input -> Internal Normalize(Scale=100), binWidth=5.0"

class MinMaxNormalizationStrategy(RadiomicsConfigStrategy):
    """
    策略：预处理阶段做了 Min-Max 归一化。
    """
    def __init__(self, target_max: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.target_max = target_max

    def get_bin_width(self) -> float:
        # 如果归一化到 [0, 1]，binWidth 必须非常小，例如 0.05 (20个bins)
        if self.target_max <= 1.0:
            return 0.05
        # 如果归一化到 [0, 1000]
        return 25.0

    def get_normalize(self) -> bool:
        return False
    
    def get_description(self) -> str:
        return f"MinMax(0-{self.target_max}) -> binWidth={self.get_bin_width()}"

# ==========================================
# 3. 策略工厂 (Factory & Registry)
# ==========================================
class StrategyFactory:
    _registry = {
        "z_score": ZScoreNormalizationStrategy,
        "raw": RawIntensityStrategy,
        "min_max": MinMaxNormalizationStrategy
    }
    
    @classmethod
    def register(cls, name: str, strategy_cls: Type[RadiomicsConfigStrategy]):
        """[扩展点] 用于注册新的自定义策略"""
        cls._registry[name] = strategy_cls
        
    @classmethod
    def get_strategy(cls, name: str, **kwargs) -> RadiomicsConfigStrategy:
        if name not in cls._registry:
            valid_keys = list(cls._registry.keys())
            raise ValueError(f"Unknown preprocessing type: '{name}'. Valid options: {valid_keys}")
        return cls._registry[name](**kwargs)

# ==========================================
# 4. 主生成函数
# ==========================================
def generate_default_config(dataset_id: str, 
                            fingerprint: dict = None, 
                            preprocessing_type: str = "z_score",
                            **strategy_kwargs) -> str:
    """
    生成 radiomics.yaml 配置文件。
    
    Args:
        dataset_id: 数据集ID
        fingerprint: 数据集指纹 (可选，可用于微调)
        preprocessing_type: 预处理策略名称 ('z_score', 'raw', 'min_max')
        **strategy_kwargs: 传递给特定策略的额外参数 (如 target_max)
    """
    
    # 1. 获取对应的策略对象
    strategy = StrategyFactory.get_strategy(preprocessing_type, **strategy_kwargs)
    
    print(f"[Config Generator] Selected Strategy: {strategy.get_description()}")
    
    # 2. 从策略中获取参数
    settings = strategy.get_setting()
    settings["binWidth"] = strategy.get_bin_width()
    settings["normalize"] = strategy.get_normalize()
    
    # 2.5 从指纹中获取 median_spacing 作为重采样目标间距
    if fingerprint and fingerprint.get("median_spacing"):
        settings["resampledPixelSpacing"] = fingerprint["median_spacing"]
        print(f"[Config Generator] resampledPixelSpacing set from fingerprint: {fingerprint['median_spacing']}")
    
    # 3. 组装完整配置
    config = {
        "setting": settings,
        "imageType": {
            "Original": {},
            "LoG": {"sigma": [1.0, 3.0, 5.0]},
            "Wavelet": {}
        },
        "featureClass": {}
    }
    
    # 4. 保存文件
    output_path = os.path.join(Radiomics_preprocessed, dataset_id, "radiomics_config.yaml")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"Configuration generated at: {output_path}")
    return output_path