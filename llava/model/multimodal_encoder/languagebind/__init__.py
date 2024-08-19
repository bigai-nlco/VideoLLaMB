import torch
from torch import nn
from transformers import AutoConfig

from .image.configuration_image import LanguageBindImageConfig
from .image.modeling_image import LanguageBindImage
from .image.tokenization_image import LanguageBindImageTokenizer
from .image.processing_image import LanguageBindImageProcessor

from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import LanguageBindVideo
from .video.tokenization_video import LanguageBindVideoTokenizer
from .video.processing_video import LanguageBindVideoProcessor

from .rmt_video.configuration_video import RMTLanguageBindVideoConfig
from .rmt_video.modeling_video import RMTLanguageBindVideo
from .rmt_video.tokenization_video import RMTLanguageBindVideoTokenizer
from .rmt_video.processing_video import RMTLanguageBindVideoProcessor

from .depth.configuration_depth import LanguageBindDepthConfig
from .depth.modeling_depth import LanguageBindDepth
from .depth.tokenization_depth import LanguageBindDepthTokenizer
from .depth.processing_depth import LanguageBindDepthProcessor

from .audio.configuration_audio import LanguageBindAudioConfig
from .audio.modeling_audio import LanguageBindAudio
from .audio.tokenization_audio import LanguageBindAudioTokenizer
from .audio.processing_audio import LanguageBindAudioProcessor

from .thermal.configuration_thermal import LanguageBindThermalConfig
from .thermal.modeling_thermal import LanguageBindThermal
from .thermal.tokenization_thermal import LanguageBindThermalTokenizer
from .thermal.processing_thermal import LanguageBindThermalProcessor

config_dict = {
    'thermal': LanguageBindThermalConfig,
    'image': LanguageBindImageConfig,
    'video': LanguageBindVideoConfig,
    'depth': LanguageBindDepthConfig,
    'audio': LanguageBindAudioConfig
}
model_dict = {
    'thermal': LanguageBindThermal,
    'image': LanguageBindImage,
    'video': LanguageBindVideo,
    'depth': LanguageBindDepth,
    'audio': LanguageBindAudio
}
transform_dict = {
    'video': LanguageBindVideoProcessor,
    'audio': LanguageBindAudioProcessor,
    'depth': LanguageBindDepthProcessor,
    'thermal': LanguageBindThermalProcessor,
    'image': LanguageBindImageProcessor,
}

class LanguageBind(nn.Module):
    def __init__(self, clip_type=('thermal', 'image', 'video', 'depth', 'audio'), use_temp=True, cache_dir='./cache_dir'):
        super(LanguageBind, self).__init__()
        self.use_temp = use_temp
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        self.modality_config = {}
        for c in clip_type:
            pretrained_ckpt = f'LanguageBind/LanguageBind_{c.capitalize()}'
            model = model_dict[c].from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
            self.modality_encoder[c] = model.vision_model
            self.modality_proj[c] = model.visual_projection
            self.modality_scale[c] = model.logit_scale
            self.modality_config[c] = model.config
        self.modality_encoder['language'] = model.text_model
        self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](**value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if self.use_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs

def to_device(x, device):
    out_dict = {k: v.to(device) for k, v in x.items()}
    return out_dict




class LanguageBindImageTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_image_tower = getattr(args, 'freeze_image_tower', True)

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindImageConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
            self.image_processor = LanguageBindImageProcessor(self.cfg_only)
            model = LanguageBindImage(self.cfg_only)
            self.image_tower = model.vision_model
    ############################################################
    def load_model(self, device_map=None):
        model = LanguageBindImage.from_pretrained(self.image_tower_name, device_map=device_map, cache_dir=self.cache_dir)
        self.image_tower = model.vision_model

        if self.freeze_image_tower:
            self.image_tower.requires_grad_(False)
        else:
            print("UNFREEZE IMAGE TOWER")

        self.image_processor = LanguageBindImageProcessor(model.config)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            # image_features = image_features[:, 1:]
            image_features = image_features.unsqueeze(1) # b 1 L D
        elif self.select_feature == 'cls_patch':
            image_features = image_features[: 1].unsqueeze(1)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # print('images', images.shape)
            image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # print('image_forward_outs', len(image_forward_outs), image_forward_outs[0].shape)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # print('image_features', image_features.shape)

        return image_features

    # # @torch.no_grad()
    # def forward_(self, images):
    #     if type(images) is list:
    #         image_features = []
    #         for image in images:
    #             image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
    #             image_feature = self.feature_select(image_forward_out).to(image.dtype)
    #             image_features.append(image_feature)
    #     else:
    #         # print('images', images.shape)
    #         image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
    #         # print('image_forward_outs', len(image_forward_outs), image_forward_outs[0].shape)
    #         image_features = self.feature_select(image_forward_outs).to(images.dtype)
    #         # print('image_features', image_features.shape)

    #     return image_features

    # def forward(self, images):
    #     if self.freeze_image_tower:
    #         with torch.no_grad():
    #             return self.forward_(images)
    #     else:
    #         return self.forward_(images)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.image_tower.embeddings.class_embedding.device  ##############

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
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class temp_model(nn.Module):
    def __init__(self):
        super(temp_model, self).__init__()
    def forward(self, **kwargs):
        return torch.randn(25, 1, 256, 1024)


class LanguageBindVideoTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_video_tower = getattr(args, 'freeze_video_tower', True)
        self.num_frames = getattr(args, 'num_frames', 8)

        # print("**********ddddbug*********")
        # print(args)
        # print("********dddbug********")

        self.cache_dir = cache_dir

        

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
            self.cfg_only.vision_config.num_frames = self.num_frames
            self.video_processor = LanguageBindVideoProcessor(self.cfg_only)
            model = LanguageBindVideo(self.cfg_only)
            self.video_tower = model.vision_model

            
    ############################################################
    def load_model(self, device_map=None):
        model_config = LanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
        model_config.vision_config.num_frames = self.num_frames
        model = LanguageBindVideo.from_pretrained(self.video_tower_name, config=model_config, device_map=device_map, cache_dir=self.cache_dir)
        self.video_processor = LanguageBindVideoProcessor(model.config)


        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.video_tower = model.vision_model
        if self.freeze_video_tower:
            self.video_tower.requires_grad_(False)
        else:
            print('UNFREEZE VIDEO TOWER')
        
        # print('=============debug==============')
        # print(model.config)
        # print('=============debug==============')

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    # def feature_select(self, video_forward_outs):
    #     # print('len(video_forward_outs.hidden_states)', len(video_forward_outs.hidden_states))
    #     video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
    #     b, t, n, c = video_features.shape
    #     # print('video_features', video_features.shape)
    #     if self.select_feature == 'patch':
    #         # video_features = video_features[:, 1:]
    #         video_features = video_features[:, :, 1:]
    #         # mean pooling
    #         video_features = video_features.mean(dim=1)
    #         video_features = video_features.reshape(b, -1, c)
    #     elif self.select_feature == 'cls_patch':
    #         # video_features = video_features
    #         video_features = video_features.reshape(b, -1, c)
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return video_features
        
    def feature_select(self, video_forward_outs):
        # print('len(video_forward_outs.hidden_states)', len(video_forward_outs.hidden_states))
        video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        b, t, n, c = video_features.shape
        # print('video_features', video_features.shape)
        if self.select_feature == 'patch':
            # get all features
            video_features = video_features
            
            # # get single frame
            # video_features = video_features[:, -1, 1:]
            
            # # get all frames
            # video_features = video_features[:, :, 1:]
            # # # mean pooling
            # # video_features = video_features.mean(1)
            # # # reshape
            # # video_features = video_features.reshape(b, -1, c)
            
        elif self.select_feature == 'cls_patch':
            # video_features = video_features
            video_features = video_features.reshape(b, -1, c)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return video_features
    
    # @torch.no_grad()
    # def forward(self, videos):
    #     if type(videos) is list:
    #         video_features = []
    #         for video in videos:
    #             video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
    #             video_feature = self.feature_select(video_forward_out).to(video.dtype)
    #             video_features.append(video_feature)
    #     else:
    #         # print(11111111111, videos.shape)
    #         video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
    #         video_features = self.feature_select(video_forward_outs).to(videos.dtype)

    #     return video_features

    # @torch.no_grad()
    def _forward(self, videos):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:
            # print(11111111111, videos.shape)
            video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

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
        return self.video_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.video_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

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
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class RMTLanguageBindVideoTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_video_tower = getattr(args, 'freeze_video_tower', True)
        self.num_frames = getattr(args, 'num_frames', 8)

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = RMTLanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
            self.cfg_only.vision_config.num_frames = self.num_frames
            self.video_processor = RMTLanguageBindVideoProcessor(self.cfg_only)
            model = RMTLanguageBindVideo(self.cfg_only)
            self.video_tower = model.vision_model
    ############################################################
    def load_model(self, device_map=None):
        model_config = RMTLanguageBindVideoConfig.from_pratrained(self.video_tower_name, cache_dir=self.cache_dir)
        model_config.vision_config.num_frames = self.num_frames
        model = RMTLanguageBindVideo.from_pretrained(self.video_tower_name, config=model_config, device_map=device_map, cache_dir=self.cache_dir)
        # model.config.vision_config = self.num_frames
        self.video_processor = RMTLanguageBindVideoProcessor(model.config)

        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.video_tower = model.vision_model
        if self.freeze_video_tower:
            self.video_tower.requires_grad_(False)
        else:
            print('UNFREEZE VIDEO TOWER')


        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    # def feature_select(self, video_forward_outs):
    #     # print('len(video_forward_outs.hidden_states)', len(video_forward_outs.hidden_states))
    #     video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
    #     b, t, n, c = video_features.shape
    #     # print('video_features', video_features.shape)
    #     if self.select_feature == 'patch':
    #         # video_features = video_features[:, 1:]
    #         video_features = video_features[:, :, 1:]
    #         # mean pooling
    #         video_features = video_features.mean(dim=1)
    #         video_features = video_features.reshape(b, -1, c)
    #     elif self.select_feature == 'cls_patch':
    #         # video_features = video_features
    #         video_features = video_features.reshape(b, -1, c)
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return video_features
        
    def feature_select(self, video_forward_outs):
        # print('len(video_forward_outs.hidden_states)', len(video_forward_outs.hidden_states))
        video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        b, n, c = video_features.shape
        # print('video_features', video_features.shape)
        if self.select_feature == 'patch':
            # get all features
            video_features = video_features
            # # video_features = video_features[:, 1:]
            # video_features = video_features[:, 1:]
            # video_features = video_features.reshape(b, -1, c)
        elif self.select_feature == 'cls_patch':
            # video_features = video_features
            video_features = video_features.reshape(b, -1, c)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return video_features

    # @torch.no_grad()
    def _forward(self, videos):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:
            # print(11111111111, videos.shape)
            video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

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
        return self.video_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.video_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

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
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2