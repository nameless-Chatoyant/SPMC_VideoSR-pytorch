import torch
from torch import nn
import torch.nn.functional as F

from math import floor, ceil

from utils import same_padding_conv, backward_warp, subpixel_upscale

class CoarseFlowEstimationConfig(object):
    ch_in = [2, 24, 24, 24, 24]
    ch_out = [24, 24, 24, 24, 32]
    kernel_size = [5, 3, 5, 3, 3]
    stride = [2, 1, 2, 1, 1]

class FineFlowEstimationConfig(object):
    ch_in = [5, 24, 24, 24, 24]
    ch_out = [24, 24, 24, 24, 8]
    kernel_size = [5, 3, 3, 3, 3]
    stride = [2, 1, 1, 1, 1]

class CoarseFlowEstimation(nn.Module):
    def __init__(self, args):
        super(CoarseFlowEstimation, self).__init__()
        self.conv_layers = [nn.Conv2d(ch_in, ch_out, kernel_size, stride) for ch_in, ch_out, kernel_size, stride in zip(args.ch_in, args.ch_out, args.kernel_size, args.stride)] 
    def forward(self, x):
        print(x.size())
        for layer_idx, conv in enumerate(self.conv_layers):
            x = same_padding_conv(x, conv)
            x = F.relu(x) if layer_idx != len(self.conv_layers) - 1 else F.tanh(x)
        x = subpixel_upscale(x, 4)
        return x

class FineFlowEstimation(nn.Module):
    def __init__(self, args):
        super(FineFlowEstimation, self).__init__()
        self.conv_layers = [nn.Conv2d(ch_in, ch_out, kernel_size, stride) for ch_in, ch_out, kernel_size, stride in zip(args.ch_in, args.ch_out, args.kernel_size, args.stride)]
    def forward(self, x):
        for layer_idx, conv in enumerate(self.conv_layers):
            x = same_padding_conv(x, conv)
            x = F.relu(x) if layer_idx != len(self.conv_layers) - 1 else F.tanh(x)
        x = subpixel_upscale(x, 2)
        return x

class MotionEstimation(nn.Module):
    def __init__(self, args):
        super(MotionEstimation, self).__init__()
        self.coarse_flow_estimation = CoarseFlowEstimation(CoarseFlowEstimationConfig())
        self.fine_flow_estimation = FineFlowEstimation(FineFlowEstimationConfig())

    def forward(self, reference, img):
        x = torch.cat([reference, img], dim = 1) # (b, 2, h, w)
        coarse_flow = self.coarse_flow_estimation(x) # (b, 2, h, w)
        sample_by_coarse_flow = backward_warping(img, coarse_flow) # (b, 1, h, w)
        x = torch.cat([reference, img, coarse_flow, sample_by_coarse_flow], dim = 1) # (b, 5, h, w)
        fine_flow = self.fine_flow_estimation(x) # (b, 2, h, w)
        
        return coarse_flow + fine_flow # (b, 2, h, w)
        
if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import numpy as np
    from scipy import misc
    me = MotionEstimation()
    # # print(me.coarse_flow, me.fine_flow)
    # img = misc.imread('data/test/1.jpg')
    # img = np.expand_dims(img, axis = 0)
    # img = img / 255.0 - 0.5
    x = Variable(torch.Tensor(np.ones((1, 1, 100, 100))))
    print(me.forward(x, x))
    # np.()
    print(me)