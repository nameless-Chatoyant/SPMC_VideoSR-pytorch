import numpy as np

def get_coords(x):
    """get coords matrix of x

    # Arguments
        x: (b, c, h, w)
    
    # Returns
        coords: (b, h, w, 2)
    """
    b, c, h, w = x.shape
    coords = np.empty((h, w, 2), dtype = np.int)
    coords[..., 0] = np.arange(h)[:, None]
    coords[..., 1] = np.arange(w)
    # coords = np.transpose(coords, (2,0,1))
    # coords = np.stack([coords] * b)

    return coords


img = np.ones((3,1,5,5))
coords = get_coords(img)
print(coords)
res = np.zeros((3,1,5,5))
print(res)
print(res[0,0,3,3])


a = np.arange(5) *2
print(coords.shape)
b = np.arange(3)
b = np.reshape(b, (-1,1,1,1))
print(b.shape)
b = np.tile(b, [1, 5, 5, 1])
print(b.shape)
# print(b)
print(coords.shape)
coords = np.stack([coords]*3, axis = 0)
print(coords.shape)
c = np.concatenate((b, coords), axis = 3)
print(c.shape)
print(c)
# print(np.choose([0,0,1],img))
print(img[0,0,1])
