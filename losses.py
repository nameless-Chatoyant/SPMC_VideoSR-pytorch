
warp_loss = 0
flow_loss = 0
loss_me = 0
euclidean_loss = 0

def me_loss_func():
    pass

def sr_loss_func():
    pass

def loss_func(stage):
    return [me_loss_func, sr_loss_func, sr_loss_func + me_loss_func][stage]