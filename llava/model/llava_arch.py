#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from einops import rearrange, repeat, pack, unpack

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_projector.self_segment import segment

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, VIDEO_TOKEN_INDEX, DEFAULT_VI_END_TOKEN, DEFAULT_VI_START_TOKEN
from llava.constants import IGNORE_INDEX, X_TOKEN_INDEX, X_INDEX_TOKEN, DEFAULT_X_PATCH_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN, X_PLACEHOLDER


from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # print('**************init***************')
        # print(config)
        # print(hasattr(config, "load_from_finetuned"))

        if hasattr(config, "mm_image_tower"):
            # print("DEBUG: load here")
            if not hasattr(config, "load_from_finetuned"):
                # print("DEBUG: load here image tower")
                self.image_tower = build_image_tower(config, delay_load=True)
                self.mm_projector = build_vision_projector(config)

                if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                    self.image_newline = nn.Parameter(
                        torch.empty(config.hidden_size, dtype=self.dtype)
                    )

        if hasattr(config, "mm_video_tower"):
            if not hasattr(config, "load_from_finetuned"):
                # print('=============ddddebug==============')
                # print(config.vision_config.num_frames)
                # print('=============ddddebug==============')
                self.video_tower = build_video_tower(config, delay_load=True)
                self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower
    
    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def initialize_image_modules(self, model_args, fsdp=None):
        image_tower = model_args.image_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer # which layer feature comes from
        mm_vision_select_feature = model_args.mm_vision_select_feature # cls_patch, patch
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        merge_mm_mlp_adapter = model_args.merge_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type # unpad, flat, spatial

        self.config.mm_image_tower = image_tower

        # image_tower = build_image_tower(model_args)

        # if fsdp is not None and len(fsdp) > 0:
        #     image_tower = self.image_tower[0]
        # else:
        #     image_tower = self.image_tower

        if self.get_image_tower() is None:
            image_tower = build_image_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.image_tower = [image_tower]
            else:
                self.image_tower = image_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                image_tower = self.image_tower[0]
            else:
                image_tower = self.image_tower
            image_tower.load_model()

        # setup config for vision_tower and project
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = image_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type # default: flat
        # config for transformer projector
        self.config.mm_layer_norm_eps = 1e-12
        self.config.mm_hidden_dropout_prob = 0.1
        self.config.mm_attention_probs_dropout_prob = 0.1
        self.config.mm_num_attention_heads = 8
        self.config.mm_intermediate_size = 4096
        self.config.mm_hidden_act = 'gelu'


        # self.mm_projector = build_vision_projector(self.config)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True


        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if 'mlp_transformer' in self.config.mm_projector_type and merge_mm_mlp_adapter is not None:
            mlp_mm_projector_weights = torch.load(merge_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {'proj.' + k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mlp_mm_projector_weights, 'mm_projector'), strict=False)

    
    def initialize_video_modules(self, model_args, fsdp=None):
        video_tower = model_args.video_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer # which layer feature comes from
        mm_vision_select_feature = model_args.mm_vision_select_feature # cls_patch, patch
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        merge_mm_mlp_adapter = model_args.merge_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type # unpad, flat, spatial

        self.config.mm_video_tower = video_tower

        # video_tower = build_video_tower(model_args)
        # print('=============debug==============')
        # print(self.get_video_tower())
        # print('=============debug==============')

        if self.get_video_tower() is None:
            video_tower = build_video_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.video_tower = [video_tower]
            else:
                self.video_tower = video_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                video_tower = self.video_tower[0]
            else:
                video_tower = self.video_tower
            video_tower.load_model()

        # config for video tower
        self.config.num_frames = getattr(model_args, 'num_frames', 8)
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = video_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type # default: flat
        # config for transformer projector
        self.config.mm_layer_norm_eps = 1e-12
        self.config.mm_hidden_dropout_prob = 0.1
        self.config.mm_attention_probs_dropout_prob = 0.1
        self.config.mm_num_attention_heads = 8
        self.config.mm_intermediate_size = 4096
        self.config.mm_hidden_act = 'gelu'

        # self.mm_projector = build_vision_projector(self.config)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if 'mlp_transformer' in self.config.mm_projector_type and merge_mm_mlp_adapter is not None:
            mlp_mm_projector_weights = torch.load(merge_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {'proj.' + k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mlp_mm_projector_weights, 'mm_projector'), strict=False)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()
    
    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def encode_images(self, images, image_sizes):
        # print("encode images")
        # original llava image encoder: flat or spatial(crop the original image to fit the off-the-shelf image processor)
        if type(images) is list or images.ndim == 5: # [B, 1, C, H, W] or [B, P, C, H, W]
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0) # B*P, C, H, W
            # image_features = self.encode_images(concat_images) # B*P, num_patches, D
            # image_features = self.model.image_tower(concat_images)
            # image_features = self.model.mm_projector(image_features)
            image_features = self.get_model().get_image_tower()(concat_images)
            image_features = self.get_model().mm_projector(image_features)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0) # B, P, num_patches, D
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat': # flatten
                image_features = [x.flatten(0, 1) for x in image_features] # [dim0, dim1] -> [dim0*dim1]
            elif mm_patch_merge_type.startswith('spatial'): # 
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features): # iterate batch
                    if image_feature.shape[0] > 1: # P, num_patches, D
                        base_image_feature = image_feature[0] # resized image
                        image_feature = image_feature[1:] # remove resized image
                        height = width = self.get_vision_tower().num_patches_per_side 
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) # (P, num_patches, D) -> (NPH, NPW, num_patches_height, num_patchs_width, D)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # (D, NPH, num_patch_height, NPW, num_patch_weight)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3) # (D, NPH*num_patch_height, NPW*num_patch_weight)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous() # (NPH, num_pathes_height, NPW, num_patches_width, D)
                            image_feature = image_feature.flatten(0, 3) # (P*num_pathes, D)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # image_features = self.model.image_tower(images)
            # image_features = self.model.mm_projector(image_features)
            image_features = self.get_model().get_image_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            # image_features = self.encode_images(images) # B, L, D

        # image_features = self.get_model().get_image_tower()(images)
        # image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_videos(self, videos, video_sizes=None):
        # print("encode videos")
        # video_features = self.model.video_tower(videos)
        # video_features = self.model.mm_projector(video_features)
        video_features = self.get_model().get_video_tower()(videos)
        # print(video_features.shape)
        video_features, all_video_features = self.get_model().mm_projector(video_features)
        return video_features
    
    def encode_image_features(self, images, image_sizes=None):
        # B T C H W -> T, N, D
        concat_images = torch.cat([image for image in images], dim=0) # B*G, C, H, W
        image_features = self.get_model().get_image_tower()(concat_images) # B*G P D
        return image_features
    
    def encode_video_features(self, videos, video_sizes=None):
        # B C T H W -> B, T, N, D
        return self.get_model().get_video_tower()(videos)
    
    
    def encode_images_retro(self, images, image_sizes):
        # print("encode images")
        # original llava image encoder: flat or spatial(crop the original image to fit the off-the-shelf image processor)
        if type(images) is list or images.ndim == 5: # [B, 1, C, H, W] or [B, P, C, H, W]
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # for image in images:
            #     print(image.shape)
            concat_images = torch.cat([image for image in images], dim=0) # B*P, C, H, W
            # image_features = self.encode_images(concat_images) # B*P, num_patches, D
            # image_features = self.model.image_tower(concat_images)
            # image_features = self.model.mm_projector(image_features)
            image_features = self.get_model().get_image_tower()(concat_images)
            image_features = self.get_model().mm_projector(image_features)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0) # B, P, num_patches, D
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat': # flatten
                image_features = [x.flatten(0, 1) for x in image_features] # [dim0, dim1] -> [dim0*dim1]
            elif mm_patch_merge_type.startswith('spatial'): # 
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features): # iterate batch
                    if image_feature.shape[0] > 1: # P, num_patches, D
                        base_image_feature = image_feature[0] # resized image
                        image_feature = image_feature[1:] # remove resized image
                        height = width = self.get_vision_tower().num_patches_per_side 
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) # (P, num_patches, D) -> (NPH, NPW, num_patches_height, num_patchs_width, D)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # (D, NPH, num_patch_height, NPW, num_patch_weight)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3) # (D, NPH*num_patch_height, NPW*num_patch_weight)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous() # (NPH, num_pathes_height, NPW, num_patches_width, D)
                            image_feature = image_feature.flatten(0, 3) # (P*num_pathes, D)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # image_features = self.model.image_tower(images)
            # image_features = self.model.mm_projector(image_features)
            image_features = self.get_model().get_image_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            # image_features = self.encode_images(images) # B, L, D

        # image_features = self.get_model().get_image_tower()(images)
        # image_features = self.get_model().mm_projector(image_features)
        # image_features = torch.stack(image_features, dim=1)
        # image_features = image_features.unsqueeze(1)
        
        image_features = torch.stack(image_features) # b, n, d
        # print("--------image--------")
        # print(image_features.shape)
        # print("--------image--------")
        # image_features = [image_features.squeeze(1)]
        image_features = [image_features]
        return image_features
    
    def encode_videos_retro(self, videos, video_sizes=None):
        
        # 1. segment for retroLM
        # # print("encode videos")
        # # video_features = self.model.video_tower(videos)
        # # video_features = self.model.mm_projector(video_features)
        # video_features = self.get_model().get_video_tower()(videos)
        # # video_features = self.get_model().mm_projector(video_features)
        # # b, t, n, d = video_features.shape
        # all_video_features = []
        # # for i in range(b):
        # # print(video_features.shape)
        # for video_feature in video_features:
        #     video_feature = video_feature.squeeze(0)
        #     # segment
        #     segs = []
        #     # print(video_feature.shape)
        #     t, n, d = video_feature.shape # t n d
        #     cls_features = video_feature[:, 1, :]
        #     video_feature = video_feature[:, 1:, :]
        #     boundaries = segment(cls_features, k=0)
        #     # print(boundaries)
        #     index = 0
        #     for bi in boundaries:
        #         steps = torch.linspace(index, bi, min(4, bi-index+1), dtype=torch.int).to(video_feature.device)
        #         seg_features = torch.index_select(video_feature, 0, steps)
        #         seg_features = seg_features.reshape(-1, d)
        #         segs.append(seg_features)
        #     segs = torch.stack(segs) # k t*n d
        #     all_video_features.append(segs)
        # all_video_features = torch.stack(all_video_features) # b, k, t*n, d
        # video_features, = self.get_model().mm_projector(all_video_features) # b, k, t*n, d
        # return video_features
        
        # 2. retro loss
        # print("encode videos")
        # video_features = self.model.video_tower(videos)
        # video_features = self.model.mm_projector(video_features)
        # videos = torch.stack(videos)
        video_features = self.get_model().get_video_tower()(videos)
        # video_features = torch.cat(video_features)
        # print(video_features.shape)
        
        all_video_features = []
        for idx, video_feature in enumerate(video_features):
            # print("--------video--------")
            # print(video_feature.shape)
            # print("--------video--------")
            _, all_video_feature = self.get_model().mm_projector(video_feature)
            if all_video_features == []:
                all_video_features = [[] for i in range(len(all_video_feature))]
            for ii, feature in enumerate(all_video_feature):
                all_video_features[ii].append(feature.squeeze(0)) # [[(1, tn, d), x4]]
        
        # video_features, all_video_features = self.get_model().mm_projector(video_features)
        # video_features = torch.stack(all_video_features, dim=1)
        # return video_features
        return all_video_features
    
        
        

    # what a awfull function: split input, get the embedding of text tokens then concatenate with visual feature, at the same time, do the truncation/padding/ignore_index
    # Thus we update the origianl bulging function
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        X, X_sizes=None, X_modalities=None
    ): # X_modalities: IMAGE, VIDEO
        
        # mixture of different modalities
        if X_modalities is None or X is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        assert len(X) == len(X_modalities)

        # # way-0: original: batch
        # x_features = getattr(self, "encode_videos")(X)
        # way-1: vanilla: iteratively encode vision feature for each item of a batch
        x_features = [getattr(self, f"encode_{X_modalities[i]}s".lower())(X[i].unsqueeze(0), [X_sizes[i]]) for i in range(len(X_modalities))]        
        x_features = [x.flatten(0, 1) for x in x_features] # [(L1*N, D), ..., (LB*N, D)], currently, we don't consider the mixture of different modalities for single input
        
        # print(len(x_features))
        # print(input_ids.shape)
        # print(X_modalities)

        # # way-2: advanced: we gather the same modality to improve the training efficiency
        # modalities = set(X_modalities) # possible shape: (B, L, C, H, W)
        # grouped_features = {m:[] for m in set(modalities)}
        # grouped_indices = {m:[] for m in set(modalities)}
        # grouped_sizes = {m:[] for m in set(modalities)}
        # for i in range(len(X_modalities)): grouped_features[X_modalities[i]].append(X[i])
        # for i in range(len(X_modalities)): grouped_indices[X_modalities[i]].append(i)
        # for i in range(len(X_modalities)): grouped_sizes[X_modalities[i]].append(X_sizes[i])
        # for k in grouped_indices: grouped_features[k] = getattr(self, f"encode_{k}s".lower())(torch.stack(grouped_features[k]), grouped_sizes[k])
        # x_features = [None] * len(X) # B, 
        # for k,v in grouped_indices.items():
        #     for i in range(len(v)): x_features[v[i]] = grouped_features[k][i] # [(L1, D), ..., (LB, D)]
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        # print(input_ids.shape)
        # print(attention_mask.shape)
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # get the embedding of the inputs tokens, then concantenate with X features
        new_input_embeds = []
        new_labels = []
        cur_x_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            CUR_X_TOKEN_INDEX = X_TOKEN_INDEX[X_modalities[batch_idx]]
            num_x = (cur_input_ids == CUR_X_TOKEN_INDEX).sum()
            # print(cur_input_ids)
            
            if num_x == 0: # pure language inputs
                # cur_x_features = x_features[cur_x_idx]
                # cur_input_embeds = torch.cat([cur_input_embeds, cur_x_features[0:0]], dim=0)
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_x_idx += 1
                continue

            # replace the special tokens with features.
            x_token_indices = [-1] + torch.where(cur_input_ids == CUR_X_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nox = []
            cur_labels = labels[batch_idx]
            cur_labels_nox = []
            for i in range(len(x_token_indices)-1) :
                cur_input_ids_nox.append(cur_input_ids[x_token_indices[i]+1:x_token_indices[i+1]])
                cur_labels_nox.append(cur_labels[x_token_indices[i]+1:x_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nox]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nox))
            # cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_nox))
            cur_input_embeds_no_x = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_x + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_x[i])
                cur_new_labels.append(cur_labels_nox[i])
                if i < num_x:
                    # print("-------")
                    # print(cur_x_idx)
                    # print(num_x)
                    # print(len(x_features))
                    # print("-------")
                    cur_x_features = x_features[cur_x_idx] # L, D
                    # print(cur_x_features.shape)
                    cur_x_idx += 1
                    cur_new_input_embeds.append(cur_x_features)
                    cur_new_labels.append(torch.full((cur_x_features.shape[0], ), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them: pad to max length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def prepare_retro_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        X, X_sizes=None, X_modalities=None
    ): # X_modalities: IMAGE, VIDEO
        
        # mixture of different modalities
        if X_modalities is None or X is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        assert len(X) == len(X_modalities)

        # print(X_modalities)

        seg_x_features = getattr(self, f"encode_{X_modalities[0]}s_retro".lower())(X, X_sizes) # b, k, t*n, d
        # num_segments = seg_x_features.shape[1]
        num_segments = len(seg_x_features)
        seg_position_ids = []
        seg_attention_mask = []
        seg_new_input_embeds = []
        seg_new_labels = []
        for segidx in range(num_segments):
            # print("*"*20)
            # print(seg_x_features.shape)
            # print("*"*20)
            # x_features = seg_x_features[:, segidx]
            x_features = seg_x_features[segidx] # [(1, t, d), ... x4]
            # print(x_features.shape)
            # print("*"*20)
            
            # TODO: image start / end is not implemented here to support pretraining.
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                raise NotImplementedError

            # Let's just add dummy tensors if they do not exist,
            # it is a headache to deal with None all the time.
            # But it is not ideal, and if you have a better idea,
            # please open an issue / submit a PR, thanks.
            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            # remove the padding using attention_mask -- FIXME
            _input_ids = input_ids
            input_ids_lst = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
            labels_lst = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

            # get the embedding of the inputs tokens, then concantenate with X features
            new_input_embeds = []
            new_labels = []
            cur_x_idx = 0
            for batch_idx, cur_input_ids in enumerate(input_ids_lst):
                CUR_X_TOKEN_INDEX = X_TOKEN_INDEX[X_modalities[batch_idx]]
                num_x = (cur_input_ids == CUR_X_TOKEN_INDEX).sum()
                if num_x == 0: # pure language inputs
                    # cur_x_features = x_features[cur_x_idx]
                    # cur_input_embeds = torch.cat([cur_input_embeds, cur_x_features[0:0]], dim=0)
                    cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                    # cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels_lst[batch_idx])
                    cur_x_idx += 1
                    continue

                # replace the special tokens with features.
                x_token_indices = [-1] + torch.where(cur_input_ids == CUR_X_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_nox = []
                cur_labels = labels_lst[batch_idx]
                cur_labels_nox = []
                for i in range(len(x_token_indices)-1) :
                    cur_input_ids_nox.append(cur_input_ids[x_token_indices[i]+1:x_token_indices[i+1]])
                    cur_labels_nox.append(cur_labels[x_token_indices[i]+1:x_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_nox]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nox))
                # cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_nox))
                cur_input_embeds_no_x = torch.split(cur_input_embeds, split_sizes, dim=0)

                cur_new_input_embeds = []
                cur_new_labels = []
                for i in range(num_x + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_x[i])
                    cur_new_labels.append(cur_labels_nox[i])
                    if i < num_x:
                        cur_x_features = x_features[cur_x_idx] # L, D
                        # print(cur_x_features.shape)
                        cur_x_idx += 1
                        cur_new_input_embeds.append(cur_x_features)
                        cur_new_labels.append(torch.full((cur_x_features.shape[0], ), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

            # Truncate sequences to max length as image embeddings can make the sequence longer
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # Combine them: pad to max length
            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            new_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        new_attention_mask[i, -cur_len:] = True
                        new_position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        new_attention_mask[i, :cur_len] = True
                        new_position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            if _labels is None:
                new_labels = None
            else:
                new_labels = new_labels_padded

            if _attention_mask is None:
                new_attention_mask = None
            else:
                new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)

            if _position_ids is None:
                new_position_ids = None

            seg_position_ids.append(new_position_ids)
            seg_attention_mask.append(new_attention_mask)
            seg_new_input_embeds.append(new_input_embeds)
            seg_new_labels.append(new_labels)

        seg_new_labels = torch.cat(seg_new_labels, dim=1)

        return None, seg_position_ids, seg_attention_mask, past_key_values, seg_new_input_embeds, seg_new_labels


    # add special tokens
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_x_patch_token:
            for x in model_args.x:
                tokenizer.add_tokens([DEFAULT_X_PATCH_TOKEN[x]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_x_start_end:
            num_new_tokens = 0
            for x in model_args.x:
                num_new_tokens += tokenizer.add_tokens([DEFAULT_X_START_TOKEN[x], DEFAULT_X_END_TOKEN[x]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_x_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
