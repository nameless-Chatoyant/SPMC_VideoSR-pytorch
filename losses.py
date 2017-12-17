import torch
from utils import get_coords, backward_warp

loss_me = 0
euclidean_loss = 0


def me_loss_func(reference, imgs, flows):
    lambda1 = 0.01
    coords = get_coords(reference)
    mappings = [coords + i for i in flows]
    warped = [backward_warp(reference, mappings[i]) for i in range(len(flows))]
    multi_reference = torch.stack([reference] * len(imgs), axis = 0)
    tensor_imgs = torch.Tensor(imgs)
    warp_loss = torch.sum(torch.abs(tensor_imgs - multi_reference))
    
    multi_flows = torch.stack(flows, dim = 0)
    flow_loss = torch.sum(torch.abs(multi_flows))

    return warp_loss + lambda1 * flow_loss

def sr_loss_func(reference_hr, outputs):
    k_range = [0.5, 1.0]
    k_range = torch.arange(*k_range, (k_range[1] - k_range[0]) / len(outputs))
    euclidean_loss = torch.sum((reference_hr - i) ** 2 for i in outputs)
    multi_euclidean_loss = torch.stack(euclidean_loss, dim = 0)
    return k_range * euclidean_loss

def loss_func(stage):
    assert stage in [1, 2, 3]
    return [me_loss_func,
        sr_loss_func,
        sr_loss_func + me_loss_func][stage - 1]