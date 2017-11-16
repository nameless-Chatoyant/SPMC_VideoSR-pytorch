from torch import nn
import torch.nn.functional as F

class CoarseFlowEstimation(nn.Module):
    def __init__(self):
        k = [5, 3, 5, 3, 3]
        n = [24, 24, 24, 24, 32]
        s = [2, 1, 2, 1, 1]
        ch_in = [2] + n[:-1]
        self.conv_layers = [nn.Conv2d(in_channels = i_ch_in,
                                        out_channels = each_n,
                                        kernel_size = each_k,
                                        stride = each_s) for i_ch_in, each_k, each_n, each_s in zip(ch_in, k, n, s)]
        
    def forward(self, x):
        for layer_idx, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.relu(x) if layer_idx != len(self.conv_layers) - 1 else F.tanh(x)
        pass

class FineFlowEstimation(nn.Module):
    def __init__(self):
        k = [5, 3, 3, 3, 3]
        n = [24, 24, 24, 24, 8]
        s = [2, 1, 1, 1, 1]
        ch_in = [2] + fine_n[:-1]
        self.conv_layers = [nn.Conv2d(in_channels = i_ch_in,
                                        out_channels = each_n,
                                        kernel_size = each_k,
                                        stride = each_s) for i_ch_in, each_k, each_n, each_s in zip(ch_in, k, n, s)]
    def forward(self, x):
        pass


class MotionEstimation(nn.Module):
    def __init__(self):
        coarse_k = [5, 3, 5, 3, 3]
        coarse_ch_in = [2, 24, 24, 24, 24]
        coarse_n = [24, 24, 24, 24, 32]
        coarse_s = [2, 1, 2, 1, 1]

        fine_ch_in = [3, 24, 24, 24, 24]
        fine_k = [5, 3, 3, 3, 3]
        fine_n = [24, 24, 24, 24, 8]
        fine_s = [2, 1, 1, 1, 1]

        self.coarse_flow = [nn.Conv2d(in_channels = ch_in,
                            out_channels = n,
                            kernel_size = k,
                            stride = s) for ch_in, k, n, s in zip(coarse_ch_in, coarse_k, coarse_n, coarse_s)]

        self.fine_flow = [nn.Conv2d(in_channels = ch_in,
                            out_channels = n,
                            kernel_size = k,
                            stride = s) for ch_in, k, n, s in zip(fine_ch_in, fine_k, fine_n, fine_s)]

    def forward(self, x):
        delta_c = x
        for i in self.coarse_flow:
            delta_c = i(delta_c)
            delta_c = F.relu(delta_c)
        print(delta_c.shape)
        quit()

        x = 0
        
if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import numpy as np
    from scipy import misc
    me = MotionEstimation()
    # print(me.coarse_flow, me.fine_flow)
    img = misc.imread('data/test/1.jpg')
    img = np.expand_dims(img, axis = 0)
    img = img / 255.0 - 0.5
    x = Variable(torch.Tensor(img))
    me.forward(x)
    # np.()