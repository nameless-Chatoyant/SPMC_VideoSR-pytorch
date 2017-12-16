import numpy as np
def get_neighbours(coords):
    """返回coords对应的neighbours，顺序为：左上、右上、左下、右下
    
    # Arguments
        coords: coords是H*W*2的矩阵，coords[v,u]的[y, x]表明原图坐标为[v,u]的像素应移动到[y,x]处
    """
    coords_lower_right = np.ceil(coords)
    coords_upper_left = np.floor(coords)
    ys_upper, xs_left = np.split(coords_upper_left, 2, axis = -1)
    ys_lower, xs_right = np.split(coords_lower_right, 2, axis = -1)
    coords_lower_left = np.concatenate((ys_lower, xs_left), axis = -1)
    coords_upper_right = np.concatenate((ys_upper, xs_right), axis = -1)
    
    return coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right

def subpixel_upscale(x, scale):
    def _subpixel_upscale(x, scale):
        b, c, h, w = x.size()
        x = np.reshape(x, (b, scale, scale, h, w))
        x = torch.split(x, h, dim = 3) # [(b, scale, scale, 1, w)] * h
        x = torch.cat([torch.squeeze(i, dim = 3) for i in x], dim = 1) # (b, scale*h, scale, w)
        x = torch.split(x, w, dim = 3) # [(b, scale*h, scale, 1)] * w
        x = torch.cat([torch.squeeze(i, dim = 3) for i in x], dim = 2) # (b, scale*h, scale*w)
        return np.reshape(x, (b, 1, scale*h, scale*w))
    x = torch.split(x, 2, dim = 1)
    x = torch.cat([_subpixel_upscale(i, scale) for i in x], dim = 1)
    return x
def forward_warp(img, mapping):
    return img
def backward_warp(img, mapping):
    return img

def get_coords(x):
    """get coords matrix of x

    # Arguments
        x: (b, c, h, w)
    
    # Returns
        coords: (b, 2, h, w)
    """
    b, c, h, w = x.size()
    coords = np.empty((h, w, 2), dtype = np.float32)
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