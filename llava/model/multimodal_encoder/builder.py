import os

from .clip_encoder import CLIPVisionTower
from .clip_vid_encoder import CLIPVideoTower
from .mae_encoder import ViTMAEVisionTower
from .vit_encoder import ViTVisionTower
from .vivit_encoder import VivitVisionTower
from .videomae_encoder import VideoMAEVisionTower

from .languagebind import LanguageBindVideoTower, LanguageBindImageTower, RMTLanguageBindVideoTower
from .deformer import DeformableImageTower
# from .egovlp import EgoVLPTower

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    # image_tower = getattr(image_tower_cfg, 'image_tower', None)
    is_absolute_path_exists = os.path.exists(image_tower)
    if "clip" in image_tower:
        if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion") or "ShareGPT4V" in image_tower:
            return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    elif "mae" in image_tower:
        if is_absolute_path_exists or image_tower.startswith("facebook"):
            return ViTMAEVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    elif "vit" in image_tower:
        if is_absolute_path_exists or image_tower.startswith("google"):
            return ViTVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    elif "LanguageBind_Image" in image_tower:
        if is_absolute_path_exists or image_tower.startswith("LanguageBind"):
            return LanguageBindImageTower(image_tower, args=image_tower_cfg, **kwargs)
    elif "deformable" in image_tower:
        if is_absolute_path_exists or image_tower.startswith("SenseTime"):
            return DeformableImageTower(image_tower, args=image_tower_cfg, **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')


def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    # video_tower = getattr(video_tower_cfg, 'video_tower', None)
    is_absolute_path_exists = os.path.exists(video_tower)
    if "clip" in video_tower:
        if is_absolute_path_exists or video_tower.startswith("openai") or video_tower.startswith("laion") or "ShareGPT4V" in video_tower:
            return CLIPVideoTower(video_tower, args=video_tower_cfg, **kwargs)
    elif "vivit" in video_tower:
        if is_absolute_path_exists or video_tower.startswith("google"):
            return VivitVisionTower(video_tower, args=video_tower_cfg, **kwargs)
    elif "videomae" in video_tower:
        if is_absolute_path_exists or video_tower.startswith("NCG-NJU"):
            return VideoMAEVisionTower(video_tower, args=video_tower_cfg, **kwargs)
    elif "RMTLanguageBind_Video_merge" in video_tower:
        if is_absolute_path_exists or video_tower.startswith("LanguageBind"):
            return RMTLanguageBindVideoTower(video_tower, args=video_tower_cfg, **kwargs)    
    elif "LanguageBind_Video_merge" in video_tower:
        if is_absolute_path_exists or video_tower.startswith("LanguageBind"):
            return LanguageBindVideoTower(video_tower, args=video_tower_cfg, **kwargs)
    # elif "EgoVLP" in video_tower:
    #     if is_absolute_path_exists or video_tower.startswith("ego"):
    #         return EgoVLPTower(video_tower, args=video_tower_cfg, **kwargs)
    

    raise ValueError(f'Unknown video tower: {video_tower}')
