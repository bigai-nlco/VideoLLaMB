import torch
import torch.nn as nn

from transformers import VivitImageProcessor, VivitModel, VivitConfig


class VivitVisionTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = VivitConfig.from_pretrained(self.video_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return

        self.video_processor = VivitImageProcessor.from_pretrained(self.video_tower_name)
        # self.video_tower = VivitModel.from_pretrained(self.video_tower_name, device_map=device_map)
        self.video_tower = VivitModel.from_pretrained(self.video_tower_name)
        self.video_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            # mean pooling
            image_features = image_features.view(image_features.shape[0], 196, image_features.shape[1]//196, image_features.shape[2])
            image_features = image_features.mean(dim=2)
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        # expected images: batch_size, num_frames, num_channels, height, width
        image_forward_outs = self.video_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

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
