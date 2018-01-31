from torch import nn
from utils import get_coords, forward_warp
class SPMC(nn.Module):
    def __init__(self):
        super(SPMC, self).__init__()

    def forward(self, img, flow, scale):
        mapping = (get_coords(img) + flow) * scale # (b, 2, h, w)
        res = forward_warp(img, mapping) # (b, 1, h*scale, w*scale)
        return res