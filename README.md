# Overview

# TODOs
- [ ] Tensorboard Support
- [ ] GIF 

```shell
SPMC_VideoSR
├── cfgs
│   ├── config_latest.py
│   └── config.py
├── data
│   └── test
│       └── 1.jpg
├── dataset.py
├── losses.py
├── model.py
├── modules
│   ├── detail_fusion_net.py
│   ├── __init__.py
│   ├── me.py
│   └── spmc.py
├── np.py
├── predict.py
├── preprocess.py
├── utils
│   ├── flow.py
│   ├── color.py
│   ├──
└── WORKBENCH.ipynb
```

# Usage
- Preprocess
- Train
- Predict
- Load model
# Results
## Image Results
|Input|Output|Groundtruth|
|:---:|:---:|:---:|
|![](imgs/input.png)|![](imgs/output.png)|![](imgs/groundtruth.png)|

## Metrics
(PSNR/SSIM)

| Method | Ours | Paper |
|:---:|:---:|:---:|
| ×2 |  | 36.71 / 0.96 |
| ×3 |  | 31.92 / 0.90 |
| ×4 |  | 29.69 / 0.84 |

#
- Our flow visualization is different from MPI-Sintel's. []() for more details about flow estimation.
- Use ffmpeg to make GIFs

# References
- Paper: [Detail-revealing Deep Video Super-resolution](https://arxiv.org/abs/1704.02738)
- Offical Repo: [jiangsutx/SPMC_VideoSR](https://github.com/jiangsutx/SPMC_VideoSR)