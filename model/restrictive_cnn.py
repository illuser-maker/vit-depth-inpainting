import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange

from .partialconv2d import PartialConv2d


class RestrictiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_shape, attention_decrease=True):
        super().__init__()
        self.ln1 = nn.LayerNorm([in_channels, *image_shape])
        self.pconv1 = PartialConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, multi_channel=False, return_mask=True, attention_decrease=True)
        self.ln2 = nn.LayerNorm([in_channels, *image_shape])
        self.pconv2 = PartialConv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, multi_channel=False, return_mask=True, attention_decrease=True)
        self.pconv3 = PartialConv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=1, bias=False, multi_channel=False, return_mask=True, attention_decrease=attention_decrease)
    
    def forward(self, input, mask=None):
        x = self.ln1(input)
        x, mask = self.pconv1(x, mask)
        x = self.ln2(x)
        x, mask = self.pconv2(x, mask)
        x = x + input
        x, mask = self.pconv3(x, mask)
        return x, mask

class RestrictiveCNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, image_shape, attention_decrease=True):
        super().__init__()
        tmp_hidden = (in_channels + hidden_channels) // 2
        self.rb1 = RestrictiveBlock(in_channels, tmp_hidden, image_shape, True)
        new_shape = [(dim)//2 + 1 for dim in image_shape]
        self.rb2 = RestrictiveBlock(tmp_hidden, hidden_channels, new_shape, True)
        new_shape = [(dim)//2 + 1 for dim in new_shape]
        self.rb3 = RestrictiveBlock(hidden_channels, hidden_channels, new_shape, True)
        new_shape = [(dim)//2 + 1 for dim in new_shape]
        self.rb4 = RestrictiveBlock(hidden_channels, out_channels, new_shape, attention_decrease)
        self.res_shape = [(dim)//2 + 1 for dim in new_shape]
    
    def forward(self, input, mask=None):
        x, mask = self.rb1(input, mask)
        x, mask = self.rb2(x, mask)
        x, mask = self.rb3(x, mask)
        x, mask = self.rb4(x, mask)
        return x, mask

class RestrictiveCNNTokenizer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, image_shape, attention_decrease):
        super().__init__()
        self.model = RestrictiveCNN(in_channels, out_channels, hidden_channels, image_shape, attention_decrease)
        self.to_tokens = Rearrange('b c h w -> b (h w) c')
        self.res_shape = self.model.res_shape

    def forward(self, input, mask=None):
        x, mask = self.model(input, mask)
        return self.to_tokens(x), self.to_tokens(mask)



        

