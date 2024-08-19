import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_image_tower = getattr(args, 'freeze_image_tower', True)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.image_tower_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.image_tower_name)
            self.image_tower = CLIPVisionModel(self.cfg_only)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.image_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_tower_name)
        # from pretrained
        self.image_tower = CLIPVisionModel.from_pretrained(self.image_tower_name, device_map=device_map)
        # # from scratch
        # image_cfg = CLIPVisionConfig.from_pretrained(self.image_tower_name)
        # self.image_tower = CLIPVisionModel(image_cfg)
        # self.image_tower.to(device_map)
        if self.freeze_image_tower:
            self.image_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # print("IMAGE: ", image_features.shape)
        if self.select_feature == 'patch':
            image_features = image_features.unsqueeze(0)
            # image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def _forward(self, images):
        
        # print("=========forward method===========")
        # print(self.dtype)
        # print("====================")

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    def forward(self, images):
        if self.freeze_image_tower:
            with torch.no_grad():
                return self._forward(images)
        else:
            return self._forward(images)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.dtype

    @property
    def device(self):
        return self.image_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
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