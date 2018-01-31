import torch
import numpy as np
from math import floor, ceil
import torch.nn.functional as F
def get_neighbours(coords):
    """返回coords对应的neighbours，顺序为：左上、右上、左下、右下
    
    # Arguments
        coords: coords是H*W*2的矩阵，coords[v,u]的[y, x]表明原图坐标为[v,u]的像素应移动到[y,x]处
    """
    coords_lr = torch.ceil(coords)
    coords_ul = torch.floor(coords)
    ys_upper, xs_left = torch.split(coords_ul, 2, axis = 1)
    ys_lower, xs_right = torch.split(coords_lr, 2, axis = 1)
    coords_ll = torch.cat((ys_lower, xs_left), axis = 1)
    coords_ur = torch.cat((ys_upper, xs_right), axis = 1)
    
    return coords_ul, coords_ur, coords_ll, coords_lr
def mapping_to_indices(coords, batch_size):
    """numpy advanced indexing is like x[<indices on axis 0>, <indices on axis 1>, ...]
        this function convert coords of shape (h, w, 2) to advanced indices
    
    # Arguments
        coords: shape of (h, w, 2)
    # Returns
        indices: [<indices on axis 0>, <indices on axis 1>, ...]
    """
    h, w = coords.shape[:2]
    indices_axis_0 = list(np.repeat(np.arange(batch_size), h * w))
    indices_axis_1 = [0]
    indices_axis_2 = list(np.tile(coords[:,:,0].reshape(-1), 2))
    indices_axis_3 = list(np.tile(coords[:,:,1].reshape(-1), batch_size))
    return [indices_axis_0, indices_axis_1, indices_axis_2, indices_axis_3]

def sample(src, coords):
    """Out of boundary coordinates will be clipped.
    """
    b, h, w, c = src.size()
    h_sample, w_sample = coords.shape[:2]
    max_coord = [h - 1, w - 1]
    coords = np.clip(coords, 0, max_coord)
    indices = mapping_to_indices(coords, b)

    sampled = src[indices].reshape((h_sample, w_sample))

    return sampled

def forward_warp(src, mapping):
    """

    # Arguments
        src: (b, h, w, c)
        mapping: from **src** to **dst**, (b, h, w, 2)
    """
    def scatter(src, coords):
        b, h, w, c = src.size()
        indices = mapping_to_indices(coords, b)
        pass
    b, h, w, c = src.size()
    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)
    diff = mapping - coords_ul
    neg_diff = 1.0 - diff
    diff_y, diff_x = torch.split(diff, 2, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 2, dim = 1)

    dst_ul = np.zeros((b, h, w, c))
    dst = np.sum([
        scatter(src * diff_y * diff_x, coords_ul),
        scatter(src * diff_y * neg_diff_x, coords_ur),
        scatter(src * neg_diff_y * diff_x, coords_ll),
        scatter(src * neg_diff_y * neg_diff_x, coords_ur)
    ], axis = -1)
    return dst

def backward_warp(src, mapping):
    """

    # Arguments
        src: source tensor, (b, h, w, c)
        mapping: from **dst** to **src**
    """
    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)

    diff = mapping - coords_ul
    neg_diff = 1.0 - diff
    diff_y, diff_x = torch.split(diff, 2, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 2, dim = 1)

    res = np.sum([
        sample(src, coords_ul) * diff_y * diff_x,
        sample(src, coords_ur) * diff_y * neg_diff_x,
        sample(src, coords_ll) * neg_diff_y * diff_x,
        sample(src, coords_ll) * neg_diff_y * neg_diff_x
    ], axis=-1)


    return src

def get_coords(h, w):
    """get coords matrix of x

    # Arguments
        h
        w
    
    # Returns
        coords: (h, w, 2)
    """
    coords = np.empty((h, w, 2), dtype = np.int)
    coords[..., 0] = np.arange(h)[:, None]
    coords[..., 1] = np.arange(w)

    return coords

def same_padding_conv(x, conv):
    dim = len(x.size())
    if dim == 4:
        b, c, h, w = x.size()
    elif dim == 5:
        b, t, c, h, w = x.size()
    else:
        raise NotImplementedError()

    if isinstance(conv, nn.Conv2d):
        padding = ((w // conv.stride[0] - 1) * conv.stride[0] + conv.kernel_size[0] - w)
        padding_l = floor(padding / 2)
        padding_r = ceil(padding / 2)
        padding = ((h // conv.stride[1] - 1) * conv.stride[1] + conv.kernel_size[1] - h)
        padding_t = floor(padding / 2)
        padding_b = ceil(padding / 2)
        x = F.pad(x, pad = (padding_l,padding_r,padding_t,padding_b))
        x = conv(x)
    elif isinstance(conv, nn.ConvTranspose2d):
        padding = ((w - 1) * conv.stride + conv.kernel_size[0] - w * conv.stride[0])
        padding_l = floor(padding / 2)
        padding_r = ceil(padding / 2)
        padding = ((h - 1) * conv.stride + conv.kernel_size[1] - h * conv.stride[1])
        padding_t = floor(padding / 2)
        padding_b = ceil(padding / 2)
        x = conv(x)
        x = x[:,:,padding_t:-padding_b,padding_l:-padding_r]
    else:
        raise NotImplementedError()
    return x


if __name__ == '__main__':
    
    # ================== Test Forward Warp ==================
    pass