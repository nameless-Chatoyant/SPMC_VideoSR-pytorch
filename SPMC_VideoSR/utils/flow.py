import numpy as np
import cv2

def read(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w = np.fromfile(f, np.int32, count = 1)[0]
            h = np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

def save(path, flow):

    magic = np.array([202021.25], np.float32)
    h, w = flow.shape[:2]
    h, w = np.array([h], np.int32), np.array([w], np.int32)

    with open(path, 'wb') as f:
        magic.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)

def flow_to_color(flow):
    """
    Hue: represents for direction
    Saturation: represents for magnitude
    Value: Keep 255
    """
    rgb_or_bgr = 'bgr'
    shape_length = len(flow.shape)
    assert rgb_or_bgr.lower() in ['rgb', 'bgr']
    assert shape_length in [3, 4]

    # following operations are based on (b, h, w, 2) ndarray
    if shape_length == 3:
        flow = np.expand_dims(flow, axis = 0)
    batch_size, img_h, img_w = flow.shape[:3]
    a = np.arctan2(-flow[:,:,:,1], -flow[:,:,:,0]) / np.pi
    h = (a + 1.0) / 2.0 * 255                       # (-1, 1) mapped to (0, 255)
    s = np.sum(flow ** 2, axis = 3) ** 0.5 * 10
    v = np.ones((batch_size, img_h, img_w)) * 255

    # build hsv image
    hsv = np.stack([h, s, v], axis = -1).astype(np.uint8)

    # hsv to rgb/bgr
    mapping = cv2.COLOR_HSV2RGB if rgb_or_bgr.lower() == 'rgb' else cv2.COLOR_HSV2BGR
    res = np.stack([cv2.cvtColor(i, mapping)] for i in hsv)

    # keep shape
    if shape_length == 3:
        res = np.squeeze(res)

    return res