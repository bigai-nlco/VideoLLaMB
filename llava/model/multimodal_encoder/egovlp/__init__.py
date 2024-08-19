
import os
import sys
import tqdm
import argparse
import numpy as np
import pdb

import torch
import torch.nn as nn

import llava.model.multimodal_encoder.egovlp.model.model as module_arch
# from .model.model import FrozenInTime
from .parse_config import ConfigParser


class EgoVLPTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False, cache_dir="./cache_dir"):
        super().__init__()

        self.is_loaded = False

        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_video_tower = getattr(args, 'freeze_video_tower', True)
        self.num_frames = getattr(args, 'num_frames', 8)

        self.cache_dir = cache_dir
        self.args = args

        

        if not delay_load:
            self.load_model()
        else:
            self.config = ConfigParser(args, test=True, eval_mode='retrotv')
            self.config._config['sliding_window_stride'] = -1
            self.video_tower = self.config.initialize('arch', module_arch)
            
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.image_tower_name))
            return

        config = ConfigParser(self.args, test=True, eval_mode="retrotv")
        config._config['sliding_window_stride'] = -1
        self.video_tower = config.initialize('arch', module_arch)
        if self.freeze_video_tower:
            self.video_tower.requires_grad_(False)
        self.is_loaded = True


    def _forward(self, videos):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                video_feature = video_forward_out.to(video.dtype)
                # video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:
            # print(11111111111, videos.shape)
            video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # video_features = self.feature_select(video_forward_outs).to(videos.dtype)
            video_feature = video_forward_outs.to(video.dtype)

        return video_features
    
    def forward(self, videos):
        if self.freeze_video_tower:
            with torch.no_grad():
                return self._forward(videos)
        else:
            return self._forward(videos)


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.video_tower.dtype

    @property
    def device(self):
        return self.video_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.video_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


if __name__ == "__main__":
    pass