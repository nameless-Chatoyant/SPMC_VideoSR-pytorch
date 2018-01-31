import torch
import numpy as np

# Details of formula are avaliable at https://en.wikipedia.org/wiki/YCbCr
# Y  = 16 + (65.481 * R + 128.553 * G + 24.966 * B)
# Cb = 128 + (-37.797 * R - 74.203 * G + 112.0 * B)
# Cr = 128 + (112.0 * R - 93.786 * G - 18.214 * B)
w = np.array([
    [65.481, 128.553, 24.966],
    [-37.797, -74.203, 112.0],
    [112.0, -93.786, -18.214]
])
w_inv = np.linalg.inv(w)
b = np.array([16, 128, 128])

def rgb_to_ycbcr(inputs, norm_func = None):
    # if inputs are normalized, perform same normalization on offset
    ib = b if norm_func is None else norm_func(b)

    # like matrix dot, w(3*3) dot RGB_pixel(3*1) = YCbCr_pixel(3*1)
    return torch.matmul(w, inputs.unsqueeze(-1)).squeeze(-1) + ib


def ycbcr_to_rgb(inputs, norm_func = None):
    # if inputs are normalized, perform same normalization on offset
    ib = b if norm_func is None else norm_func(b)

    # like matrix dot, w(3*3) dot YCbCr_pixel(3*1) = RGB_pixel(3*1)
    return torch.matmul(w_inv, (inputs - ib).unsqueeze(-1)).squeeze(-1)