from .restrictive_cnn import RestrictiveCNNTokenizer
from .vit import ViT
from .decoder import Decoder

import torch
from torch import nn
from einops.layers.torch import Rearrange

class CoarseNet(nn.Module):
    def __init__(self, token_channels, token_hidden, image_shape, depth=3, heads=16):
        super().__init__()
        # RGB
        self.tokenizer1 = RestrictiveCNNTokenizer(in_channels=3, out_channels=token_channels, hidden_channels=token_hidden, image_shape=image_shape[-2:], attention_decrease=False)
        res_shape = 256
        self.transformer_encoder1 = ViT(image_size = image_shape, patch_size = 32, out_dim = 256, 
            dim = 128, depth = depth, heads = heads, mlp_dim = 2048, patch_model = True, patch_num = res_shape,
            dropout = 0.1, emb_dropout = 0.1, masked=True)
        # Depth4
        self.tokenizer2 = RestrictiveCNNTokenizer(in_channels=1, out_channels=token_channels, hidden_channels=token_hidden, image_shape=image_shape[-2:], attention_decrease=True)
        self.transformer_encoder2 = ViT(image_size = image_shape, patch_size = 32, out_dim = 256,
            dim = 128, depth = depth, heads = heads, mlp_dim = 2048, patch_model = True, patch_num = res_shape,
            dropout = 0.1, emb_dropout = 0.1, masked=True)
        
        self.decoder = Decoder(256+256)
        self.unflatten = Rearrange('b (h w) c -> b c h w', h=14, w=14)

    def forward(self, input_rgb, input_dep, mask=None):
        rgb_tokens, _ = self.tokenizer1(input_rgb)
        dep_tokens, mask = self.tokenizer2(input_dep, mask)
        features_rgb = self.transformer_encoder1(rgb_tokens, mask)
        features_dep = self.transformer_encoder2(dep_tokens, mask)
        total_features = torch.cat((features_rgb, features_dep), dim=-1)
        output = self.decoder(self.unflatten(total_features))
        if not self.training:
            output = torch.clamp(output, min=0)
        return output