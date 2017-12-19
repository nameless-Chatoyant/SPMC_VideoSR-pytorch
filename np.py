import numpy as np
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
def get_neighbours(coords):
    """返回coords对应的neighbours，顺序为：左上、右上、左下、右下
    
    # Arguments
        coords: coords是H*W*2的矩阵，coords[v,u]的[y, x]表明原图坐标为[v,u]的像素应移动到[y,x]处
    """
    coords_lower_right = np.ceil(coords).astype(np.int)
    coords_upper_left = np.floor(coords).astype(np.int)
    ys_upper, xs_left = np.split(coords_upper_left, 2, axis = -1)
    ys_lower, xs_right = np.split(coords_lower_right, 2, axis = -1)
    coords_lower_left = np.concatenate((ys_lower, xs_left), axis = -1)
    coords_upper_right = np.concatenate((ys_upper, xs_right), axis = -1)
    
    return coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right
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

# numpy advanced indexing is like x[[indices on axis 0], [indices on axis 1], ...]
# following is converting coords of shape (h, w, 2) to advanced indices of shape (4,)
coor = get_coords(3, 3)
mapping = coor + 0.5
coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right = get_neighbours(mapping)

indices_upper_left = mapping_to_indices(coords_upper_left, 2)
indices_upper_right = mapping_to_indices(coords_upper_right, 2)
indices_lower_left = mapping_to_indices(coords_lower_left, 2)
indices_lower_right = mapping_to_indices(coords_lower_right, 2)
indices = mapping_to_indices(mapping, 2)
print(*indices, sep = '\n')
print('--------------------------------------')
print(*indices_upper_left, sep = '\n')

a = np.ones((2,1,3,3))
print('before:', a, sep = '\n')
b = np.zeros((2,1,3,3))
# a[indices] = 0
b = a[indices_upper_left] + a[indices_upper_right] + a[indices_lower_left] + a[indices_lower_right]
print(b)
# print('after:', a, sep = '\n')