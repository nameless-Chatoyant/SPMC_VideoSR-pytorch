
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


def forward_warping(img, coords):
    pass

def backward_warping(img, coords):
    pass

def sub_pixel_upscale():
    pass