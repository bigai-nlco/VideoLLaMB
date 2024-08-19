import torch
import torch.nn as nn
import re

from .identity_projector import IdentityProjector
from .mlp_projector import MLPProjector
from .qformer_projector import qformer_config_template, Blip2Model
from .transformer_projector import TransformerProjector
from .mlp_transformer_projector import MLPTransformerProjector
from .rmt_transformer_projector import RMTTransformerProjector
from .rmt_r_transformer_projector import RMTRTransformerProjector

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # three types of projector: linear, typical FFN, identity

    if projector_type == 'identity':
        return IdentityProjector()
    elif projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'pool':
        pass # TODO
    elif 'mlp_transformer' in projector_type:
        mlp_transformer_layer = re.match(r'^mlp_transformer(\d+)x', projector_type)
        mlp_transformer_depth = int(mlp_transformer_layer.group(1))
        return MLPTransformerProjector(config, mlp_transformer_depth)
    elif 'rmt_transformer' in projector_type:
        rmt_transformer_layer = re.match(r'^rmt_transformer(\d+)x', projector_type)
        rmt_transformer_depth = int(rmt_transformer_layer.group(1))
        return RMTTransformerProjector(config, rmt_transformer_depth)
    elif 'rmt_r_transformer' in projector_type:
        rmt_r_transformer_layer = re.match(r'^rmt_r_transformer(\d+)x', projector_type)
        rmt_r_transformer_depth = int(rmt_r_transformer_layer.group(1))
        return RMTRTransformerProjector(config, rmt_r_transformer_depth)
    elif 'transformer' in projector_type:
        transformer_layer = re.match(r'^transformer(\d+)x', projector_type)
        transformer_depth = int(transformer_layer.group(1))
        return TransformerProjector(config, transformer_depth)
    elif 'mlp' in projector_type:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
        # return MLPProjector(config, mlp_depth)
    elif 'qformer' in projector_type:
        qformer_config = qformer_config_template(config, projector_type)
        return Blip2Model(qformer_config)

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_resampler(config, delay_load=False, **kwargs):
    # refer to qformer in blip2 or resampler in flanmingo
    pass # TODO

def build_vision_vqgan(config, delay_load=False, **kwargs):
    # refer to lavit
    pass # TODO
