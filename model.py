import torch
from torch import nn
import torch.nn.functional as F
from modules import *
from cfgs.config import args
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.me = MotionEstimation(args.motion_estimation)
        self.spmc = SPMC()
        self.detail_fusion_net = DetailFusionNet(args.detail_fusion_net)
    def forward(self, hr_sparses, lr):
        """
        # Arguments
            hr_sparses: [(b, c, h*scale, w*scale)] *t
            lr: (b, c, h, w)
        """
        for i in hr_sparses:
            feature = self.encoder()
            print(self.encoder.skip_connections)
        return feature