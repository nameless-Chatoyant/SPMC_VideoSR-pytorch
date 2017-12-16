import torch
from torch import nn
import torch.nn.functional as F
from utils import get_coords, forward_warp
class SPMC(nn.Module):
    def __init__(self):
        super(SPMC, self).__init__()

    def forward(self, img, flow, scale):
        coords = get_coords(img) # (b, 2, h, w)
        mapping = (coords + flow) * scale # (b, 2, h, w)
        res = forward_warp(img, mapping) # (b, 1, h*scale, w*scale)
        return res