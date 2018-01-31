
from torch import nn
from easydict import EasyDict as edict

args = edict(
    motion_estimation = edict(),
    detail_fusion_net = edict()
)

args.motion_estimation.coarse_flow_estimation = edict(
    ch_in = [2, 24, 24, 24, 24],
    ch_out = [24, 24, 24, 24, 32],
    kernel_size = [5, 3, 5, 3, 3],
    stride = [2, 1, 2, 1, 1]
)

args.motion_estimation.fine_flow_estimation = edict(
    ch_in = [5, 24, 24, 24, 24],
    ch_out = [24, 24, 24, 24, 8],
    kernel_size = [5, 3, 3, 3, 3],
    stride = [2, 1, 1, 1, 1]    
)

args.detail_fusion_net.encoder = edict(
    ch_in = [1, 32, 64, 64],
    ch_out = [32, 64, 64, 128],
    kernel_size = [5, 3, 3, 3],
    stride = [1, 2, 1, 2]
)

args.detail_fusion_net.decoder = edict(
    ch_in = [128, 128, 64, 64, 32, 32],
    ch_out = [128, 64, 64, 32, 32, 1],
    kernel_size = [3, 4, 3, 4, 3, 5],
    stride = [1, 2, 1, 2, 1, 1],
    types = [nn.Conv2d, nn.ConvTranspose2d, nn.Conv2d, nn.ConvTranspose2d, nn.Conv2d, nn.Conv2d]
)

if __name__ == '__main__':
    print(args)